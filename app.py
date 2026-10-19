"""HTTP API for the two-stage CF challenge screenshot detector."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
import gc
import io
import logging
import os
from pathlib import Path
import secrets
import threading
import time
import uuid
import warnings

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from cfbox_detector import CFChallengeDetector

# Keep the original opener. The torch backend imports Ultralytics, which
# globally patches PIL.Image.open to try pi-heif for unknown files.
_PIL_OPEN = Image.open

SERVICE_VERSION = '0.1.0'
logger = logging.getLogger('cfbox.service')


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f'{name} must be an integer') from exc
    if value < minimum:
        raise RuntimeError(f'{name} must be >= {minimum}')
    return value


def _env_float(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f'{name} must be a number') from exc
    if not low <= value <= high:
        raise RuntimeError(f'{name} must be between {low} and {high}')
    return value


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name, 'true' if default else 'false').strip().lower()
    if value in {'1', 'true', 'yes', 'on'}:
        return True
    if value in {'0', 'false', 'no', 'off'}:
        return False
    raise RuntimeError(f'{name} must be a boolean')


@dataclass(frozen=True)
class Settings:
    widget_weights: Path
    state_weights: Path
    manifest: Path | None
    backend: str
    threads: int
    conf_widget: float
    conf_state: float
    conflict_margin: float
    iou: float
    imgsz_widget: int
    imgsz_state: int
    max_upload_bytes: int
    max_pixels: int
    api_key: str | None
    device: str
    log_level: str
    idle_unload_seconds: int
    load_on_startup: bool

    @classmethod
    def from_env(cls) -> 'Settings':
        upload_mb = _env_int('CFBOX_MAX_UPLOAD_MB', 10)
        backend = os.environ.get('CFBOX_BACKEND', 'onnx').strip().lower()
        if backend not in {'auto', 'onnx', 'torch'}:
            raise RuntimeError('CFBOX_BACKEND must be auto, onnx, or torch')
        manifest = os.environ.get('CFBOX_MODEL_MANIFEST',
                                  '/models/onnx/manifest.json').strip()
        return cls(
            widget_weights=Path(os.environ.get('CFBOX_WIDGET_WEIGHTS',
                                               '/models/onnx/widget.onnx')),
            state_weights=Path(os.environ.get('CFBOX_STATE_WEIGHTS',
                                              '/models/onnx/state.onnx')),
            manifest=Path(manifest) if manifest else None,
            backend=backend,
            threads=_env_int('CFBOX_THREADS', 8),
            conf_widget=_env_float('CFBOX_CONF_WIDGET', 0.25, 0, 1),
            conf_state=_env_float('CFBOX_CONF_STATE', 0.25, 0, 1),
            conflict_margin=_env_float('CFBOX_CONFLICT_MARGIN', 0.15, 0, 1),
            iou=_env_float('CFBOX_IOU', 0.7, 0, 1),
            imgsz_widget=_env_int('CFBOX_IMGSZ_WIDGET', 1280),
            imgsz_state=_env_int('CFBOX_IMGSZ_STATE', 640),
            max_upload_bytes=upload_mb * 1024 * 1024,
            max_pixels=_env_int('CFBOX_MAX_PIXELS', 50_000_000),
            api_key=os.environ.get('CFBOX_API_KEY', '').strip() or None,
            device=os.environ.get('CFBOX_DEVICE', 'cpu').strip() or 'cpu',
            log_level=os.environ.get('CFBOX_LOG_LEVEL', 'INFO').upper(),
            idle_unload_seconds=_env_int('CFBOX_IDLE_UNLOAD_SECONDS', 300, 0),
            load_on_startup=_env_bool('CFBOX_LOAD_ON_STARTUP', True),
        )


def _configure_logging(level: str) -> None:
    level_value = getattr(logging, level, logging.INFO)
    logging.basicConfig(level=level_value,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger.setLevel(level_value)


def _require_api_key(settings: Settings, x_api_key: str | None,
                     authorization: str | None) -> None:
    if not settings.api_key:
        return
    token = (x_api_key or '').strip()
    if not token and authorization and authorization.lower().startswith('bearer '):
        token = authorization[7:].strip()
    if not token or not secrets.compare_digest(token, settings.api_key):
        raise HTTPException(status_code=401,
                            detail={'code': 'unauthorized',
                                    'message': 'A valid API key is required'},
                            headers={'WWW-Authenticate': 'Bearer'})


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    chunks = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413,
                                detail={'code': 'image_too_large',
                                        'message': f'Upload exceeds {limit} bytes'})
        chunks.append(chunk)
    if not total:
        raise HTTPException(status_code=400,
                            detail={'code': 'empty_image',
                                    'message': 'Uploaded image is empty'})
    return b''.join(chunks)


def _decode_image(data: bytes, max_pixels: int) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with _PIL_OPEN(io.BytesIO(data)) as opened:
                if opened.width * opened.height > max_pixels:
                    raise HTTPException(status_code=400,
                                        detail={'code': 'image_too_large',
                                                'message': 'Image dimensions exceed the limit'})
                image = ImageOps.exif_transpose(opened).convert('RGB')
                image.load()
                return image
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise HTTPException(status_code=400,
                            detail={'code': 'invalid_image',
                                    'message': 'Upload is not a readable image'}) from exc


def _clear_model_state(app: FastAPI, reason: str) -> None:
    app.state.detector = None
    app.state.ready = False
    app.state.last_used = 0.0
    app.state.last_unload_reason = reason
    app.state.model_unloads += 1


def _release_memory() -> None:
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _load_models_sync(app: FastAPI) -> bool:
    """Load models if absent. Returns True when a new load happened."""
    with app.state.model_lock:
        if app.state.detector is not None:
            app.state.last_used = time.monotonic()
            return False
        settings = app.state.settings
        if settings.backend == 'onnx':
            imgsz_widget = imgsz_state = None
        else:
            imgsz_widget, imgsz_state = settings.imgsz_widget, settings.imgsz_state
        try:
            detector = CFChallengeDetector(
                settings.widget_weights, settings.state_weights,
                imgsz_widget=imgsz_widget, imgsz_state=imgsz_state,
                conf_widget=settings.conf_widget, conf_state=settings.conf_state,
                conflict_margin=settings.conflict_margin, iou=settings.iou,
                backend=settings.backend, manifest=settings.manifest,
                device=settings.device, threads=settings.threads)
        except Exception:
            app.state.ready = False
            raise
        app.state.detector = detector
        app.state.ready = True
        app.state.loaded_at = time.time()
        app.state.last_used = time.monotonic()
        app.state.model_loads += 1
        return True


def _unload_models_sync(app: FastAPI, reason: str) -> bool:
    with app.state.model_lock:
        if app.state.detector is None:
            return False
        _clear_model_state(app, reason)
    _release_memory()
    return True


def _unload_if_idle_sync(app: FastAPI, timeout: int) -> bool:
    with app.state.model_lock:
        if app.state.detector is None:
            return False
        if time.monotonic() - app.state.last_used < timeout:
            return False
        _clear_model_state(app, 'idle')
    _release_memory()
    return True


async def _idle_monitor(app: FastAPI) -> None:
    timeout = app.state.settings.idle_unload_seconds
    if timeout <= 0:
        return
    interval = min(5.0, max(0.5, timeout / 4))
    while True:
        await asyncio.sleep(interval)
        removed = await run_in_threadpool(_unload_if_idle_sync, app, timeout)
        if removed:
            logger.info('models unloaded after %ss idle', timeout)


def _predict_sync(app: FastAPI, image: Image.Image) -> dict:
    with app.state.model_lock:
        _load_models_sync(app)
        try:
            return app.state.detector.predict(image)
        finally:
            app.state.last_used = time.monotonic()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    _configure_logging(settings.log_level)
    Image.MAX_IMAGE_PIXELS = settings.max_pixels
    app.state.settings = settings
    app.state.started_at = time.time()
    app.state.ready = False
    app.state.detector = None
    app.state.model_lock = threading.RLock()
    app.state.loaded_at = None
    app.state.last_used = 0.0
    app.state.model_loads = 0
    app.state.model_unloads = 0
    app.state.last_unload_reason = None
    app.state.idle_task = None
    logger.info('loading backend=%s widget=%s state=%s threads=%s',
                settings.backend, settings.widget_weights, settings.state_weights,
                settings.threads)
    if settings.load_on_startup:
        try:
            await run_in_threadpool(_load_models_sync, app)
        except Exception:
            logger.exception('failed to load detector models')
            raise
        logger.info('service ready version=%s', SERVICE_VERSION)
    else:
        logger.info('startup model load disabled; use POST /models/load or /predict')
    if settings.idle_unload_seconds > 0:
        app.state.idle_task = asyncio.create_task(_idle_monitor(app))
    try:
        yield
    finally:
        task = app.state.idle_task
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        await run_in_threadpool(_unload_models_sync, app, 'shutdown')


app = FastAPI(title='CFbox Detector API', version=SERVICE_VERSION,
              description='Two-stage CF challenge screenshot detection.',
              lifespan=lifespan)


@app.get('/')
def root():
    return {'service': 'cfbox-detector', 'version': SERVICE_VERSION,
            'docs': '/docs', 'health': '/health', 'predict': '/predict',
            'models_load': '/models/load', 'models_unload': '/models/unload'}


@app.get('/health')
def health(request: Request):
    return {'status': 'ok', 'version': SERVICE_VERSION,
            'ready': bool(getattr(request.app.state, 'ready', False))}


@app.get('/ready')
def ready(request: Request):
    if not getattr(request.app.state, 'ready', False):
        return JSONResponse(status_code=503,
                            content={'status': 'unloaded', 'ready': False,
                                     'loaded': False})
    return {'status': 'ready', 'ready': True, 'loaded': True}


@app.get('/info')
def info(request: Request):
    settings = getattr(request.app.state, 'settings', None)
    if settings is None:
        return JSONResponse(status_code=503,
                            content={'status': 'loading', 'ready': False})
    loaded = request.app.state.detector is not None
    last_used = request.app.state.last_used
    loaded_at = request.app.state.loaded_at
    return {
        'service': 'cfbox-detector',
        'version': SERVICE_VERSION,
        'ready': bool(request.app.state.ready),
        'loaded': loaded,
        'uptime_seconds': round(time.time() - request.app.state.started_at, 1),
        'backend': settings.backend,
        'models': {'widget': str(settings.widget_weights),
                   'state': str(settings.state_weights),
                   'manifest': str(settings.manifest) if settings.manifest else None},
        'thresholds': {'conf_widget': settings.conf_widget,
                       'conf_state': settings.conf_state,
                       'conflict_margin': settings.conflict_margin,
                       'iou': settings.iou},
        'limits': {'max_upload_bytes': settings.max_upload_bytes,
                   'max_pixels': settings.max_pixels,
                   'threads': settings.threads},
        'lifecycle': {
            'load_on_startup': settings.load_on_startup,
            'idle_unload_seconds': settings.idle_unload_seconds,
            'loaded_seconds': (round(time.time() - loaded_at, 1)
                               if loaded and loaded_at else None),
            'last_used_seconds_ago': (round(time.monotonic() - last_used, 1)
                                      if loaded and last_used else None),
            'model_loads': request.app.state.model_loads,
            'model_unloads': request.app.state.model_unloads,
            'last_unload_reason': request.app.state.last_unload_reason,
        },
        'auth_required': bool(settings.api_key),
    }


@app.post('/models/load')
async def load_models(request: Request,
                      x_api_key: str | None = Header(default=None, alias='X-API-Key'),
                      authorization: str | None = Header(default=None)):
    settings: Settings = request.app.state.settings
    _require_api_key(settings, x_api_key, authorization)
    started = time.perf_counter()
    try:
        loaded = await run_in_threadpool(_load_models_sync, request.app)
    except Exception as exc:
        logger.exception('model load failed')
        raise HTTPException(status_code=500,
                            detail={'code': 'model_load_failed',
                                    'message': 'Failed to load models'}) from exc
    return {'status': 'loaded', 'loaded': True, 'already_loaded': not loaded,
            'load_ms': round((time.perf_counter() - started) * 1000, 1)}


@app.post('/models/unload')
async def unload_models(request: Request,
                        x_api_key: str | None = Header(default=None, alias='X-API-Key'),
                        authorization: str | None = Header(default=None)):
    settings: Settings = request.app.state.settings
    _require_api_key(settings, x_api_key, authorization)
    removed = await run_in_threadpool(_unload_models_sync, request.app, 'api')
    return {'status': 'unloaded', 'loaded': False,
            'already_unloaded': not removed}


@app.post('/predict')
async def predict(request: Request,
                  image: UploadFile = File(..., description='Screenshot to inspect'),
                  x_api_key: str | None = Header(default=None, alias='X-API-Key'),
                  authorization: str | None = Header(default=None)):
    settings: Settings = request.app.state.settings
    _require_api_key(settings, x_api_key, authorization)
    request_id = uuid.uuid4().hex
    started = time.perf_counter()
    data = await _read_limited(image, settings.max_upload_bytes)
    screenshot = _decode_image(data, settings.max_pixels)
    try:
        result = await run_in_threadpool(_predict_sync, request.app, screenshot)
    except Exception as exc:
        logger.exception('inference failed request_id=%s', request_id)
        raise HTTPException(status_code=500,
                            detail={'code': 'inference_failed',
                                    'message': 'Inference failed'}) from exc
    duration_ms = (time.perf_counter() - started) * 1000
    payload = {'request_id': request_id, **result}
    logger.info('prediction request_id=%s state=%s has_cf=%s size=%sx%s duration_ms=%.1f',
                request_id, result['state'], result['has_cf_challenge'],
                screenshot.width, screenshot.height, duration_ms)
    return JSONResponse(payload,
                        headers={'X-Request-ID': request_id,
                                 'Cache-Control': 'no-store'})
