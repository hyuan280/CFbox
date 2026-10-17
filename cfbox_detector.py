"""Two-stage CF challenge screenshot detector.

Stage 1 locates the challenge widget in the full screenshot. Stage 2 classifies
the widget crop and, for the unchecked state, locates the checkbox.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics'))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')

MODEL_STATE_NAMES = {
    'checkbox_unchecked': 'unchecked',
    'state_verified': 'verified',
    'state_processing': 'processing',
    'state_error': 'error',
}
RESULT_STATES = ('absent', 'unchecked', 'verified', 'processing', 'error', 'unknown')
POSITIVE_STATES = ('unchecked', 'verified', 'processing', 'error')


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    box: tuple[float, float, float, float]


def _round(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def box_center(box: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def expand_box(box: Sequence[float], margin: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (max(0, int(math.floor(x1)) - margin),
            max(0, int(math.floor(y1)) - margin),
            min(width, int(math.ceil(x2)) + margin),
            min(height, int(math.ceil(y2)) + margin))


def box_to_pct(box: Sequence[float], width: int, height: int) -> dict[str, float]:
    x1, y1, x2, y2 = box
    return {'x1': _round(x1 / width * 100), 'y1': _round(y1 / height * 100),
            'x2': _round(x2 / width * 100), 'y2': _round(y2 / height * 100)}


def point_in_box(point: Sequence[float], box: Sequence[float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def valid_checkbox_box(box: Sequence[float], aspect_range: tuple[float, float] = (0.5, 2.0),
                       min_side: float = 4.0) -> bool:
    """Reject thin strips and tiny boxes before treating a detection as a checkbox."""
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    if width < min_side or height < min_side:
        return False
    aspect = width / height
    return aspect_range[0] <= aspect <= aspect_range[1]


def resolve_state(detections: Iterable[Detection], state_by_class: dict[int, str],
                  conflict_margin: float) -> tuple[str, Detection | None, str | None]:
    """Pick one state, rejecting ambiguous class conflicts."""
    eligible = sorted((d for d in detections if d.class_id in state_by_class),
                      key=lambda d: d.confidence, reverse=True)
    if not eligible:
        return 'unknown', None, 'no_state_detection'
    top = eligible[0]
    for other in eligible[1:]:
        if other.class_id != top.class_id and other.confidence >= top.confidence - conflict_margin:
            return 'unknown', None, 'conflicting_state_detections'
    return state_by_class[top.class_id], top, None


def annotate_result(image: Image.Image, result: dict[str, Any]) -> Image.Image:
    canvas = image.convert('RGB').copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    colors = {'unchecked': '#d97706', 'verified': '#15803d', 'processing': '#2563eb',
              'error': '#b91c1c', 'unknown': '#6b7280', 'absent': '#6b7280'}
    challenges = result.get('challenges') or []
    if not challenges:
        draw.rectangle((8, 8, 108, 30), fill='#111827')
        draw.text((14, 13), 'absent', fill='white')
        return canvas
    for challenge in challenges:
        color = colors.get(challenge.get('state'), colors['unknown'])
        box = challenge.get('widget_box_px')
        if box:
            draw.rectangle(tuple(box), outline=color, width=3)
            label = f"{challenge.get('state')} {challenge.get('confidence') or 0:.2f}"
            tx, ty = box[0] + 3, max(0, box[1] - 15)
            draw.rectangle((tx - 2, ty - 1, tx + 8 * len(label) + 4, ty + 13), fill='#111827')
            draw.text((tx, ty), label, fill='white')
        center = challenge.get('checkbox_center_pct')
        if center:
            cx, cy = center['x'] / 100 * width, center['y'] / 100 * height
            draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), outline=color, width=3)
            draw.line((cx - 11, cy, cx + 11, cy), fill=color, width=2)
            draw.line((cx, cy - 11, cx, cy + 11), fill=color, width=2)
    return canvas


class CFChallengeDetector:
    def __init__(self, widget_weights: str | Path = ROOT / 'runs/widget/weights/best.pt',
                 state_weights: str | Path = ROOT / 'runs/state/weights/best.pt',
                 *, imgsz_widget: int = 1280, imgsz_state: int = 640,
                 conf_widget: float = 0.25, conf_state: float = 0.25,
                 device: str = 'cpu', threads: int = 8, max_det: int = 20,
                 crop_margin: int = 8, conflict_margin: float = 0.15,
                 checkbox_aspect_range: tuple[float, float] = (0.5, 2.0)):
        if not 0 <= conf_widget <= 1 or not 0 <= conf_state <= 1:
            raise ValueError('Confidence thresholds must be between 0 and 1')
        if not 0 <= conflict_margin <= 1:
            raise ValueError('conflict_margin must be between 0 and 1')
        if min(imgsz_widget, imgsz_state, threads, max_det) < 1 or crop_margin < 0:
            raise ValueError('Sizes and thread count must be positive')
        if not 0 < checkbox_aspect_range[0] <= checkbox_aspect_range[1]:
            raise ValueError('checkbox_aspect_range must be positive and ordered')
        widget_path, state_path = Path(widget_weights), Path(state_weights)
        for path in (widget_path, state_path):
            if not path.is_file():
                raise FileNotFoundError(f'Weights not found: {path}')
        import torch
        from ultralytics import YOLO, settings
        settings.update({'sync': False})
        torch.set_num_threads(int(threads))
        self.widget_model = YOLO(str(widget_path))
        self.state_model = YOLO(str(state_path))
        names = {int(k): str(v) for k, v in self.widget_model.names.items()}
        widget_ids = [idx for idx, name in names.items() if name == 'cf_widget']
        if len(widget_ids) != 1:
            raise ValueError('Widget model must contain exactly one cf_widget class')
        self.widget_class = widget_ids[0]
        self.state_by_class = {}
        for class_id, name in self.state_model.names.items():
            if name not in MODEL_STATE_NAMES:
                raise ValueError(f'Unexpected state class: {name}')
            self.state_by_class[int(class_id)] = MODEL_STATE_NAMES[name]
        if set(self.state_by_class.values()) != set(MODEL_STATE_NAMES.values()):
            raise ValueError('State model must contain all four state classes')
        self.imgsz_widget = imgsz_widget
        self.imgsz_state = imgsz_state
        self.conf_widget = conf_widget
        self.conf_state = conf_state
        self.device = device
        self.max_det = max_det
        self.crop_margin = crop_margin
        self.conflict_margin = conflict_margin
        self.checkbox_aspect_range = checkbox_aspect_range

    @staticmethod
    def _as_image(source: str | Path | Image.Image | Any) -> Image.Image:
        if isinstance(source, (str, Path)):
            with Image.open(source) as image:
                return image.convert('RGB')
        if isinstance(source, Image.Image):
            return source.convert('RGB')
        try:
            return Image.fromarray(source).convert('RGB')
        except Exception as exc:
            raise TypeError('Image must be a path, PIL image, or numpy array') from exc

    def _predict_boxes(self, image: Image.Image, model: Any, imgsz: int, conf: float,
                       classes: list[int]) -> list[Detection]:
        result = model.predict(source=image, imgsz=imgsz, conf=conf, iou=0.7,
                               device=self.device, verbose=False,
                               max_det=self.max_det, augment=False,
                               classes=classes)[0]
        detections = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            detections.append(Detection(int(box.cls[0].item()), float(box.conf[0].item()),
                                        tuple(float(v) for v in box.xyxy[0].tolist())))
        return detections

    def predict_state(self, image: str | Path | Image.Image) -> dict[str, Any]:
        """Run only the state model on an already-cropped widget image."""
        crop = self._as_image(image)
        started = time.perf_counter()
        detections = self._predict_boxes(crop, self.state_model, self.imgsz_state,
                                         self.conf_state, sorted(self.state_by_class))
        detections = [d for d in detections
                      if self.state_by_class[d.class_id] != 'unchecked'
                      or valid_checkbox_box(d.box, self.checkbox_aspect_range)]
        state, chosen, reason = resolve_state(detections, self.state_by_class,
                                              self.conflict_margin)
        result: dict[str, Any] = {
            'state': state,
            'confidence': _round(chosen.confidence, 4) if chosen else None,
            'state_box_px': [_round(v) for v in chosen.box] if chosen else None,
            'reason': reason,
            'timing_ms': _round((time.perf_counter() - started) * 1000),
        }
        if state == 'unchecked' and chosen:
            result['checkbox_center_px'] = [_round(v) for v in box_center(chosen.box)]
        return result

    def predict(self, image: str | Path | Image.Image) -> dict[str, Any]:
        source = self._as_image(image)
        width, height = source.size
        started = time.perf_counter()
        widget_detections = self._predict_boxes(source, self.widget_model,
                                                self.imgsz_widget, self.conf_widget,
                                                [self.widget_class])
        widget_ms = (time.perf_counter() - started) * 1000
        if not widget_detections:
            return {
                'has_cf_challenge': False,
                'state': 'absent',
                'confidence': None,
                'checkbox_center_pct': None,
                'ambiguous': False,
                'reason': None,
                'challenges': [],
                'image_size': [width, height],
                'timing_ms': {'widget': _round(widget_ms),
                              'state': 0.0,
                              'total': _round((time.perf_counter() - started) * 1000)},
            }
        challenges = []
        state_ms = 0.0
        for detection in sorted(widget_detections, key=lambda d: d.confidence, reverse=True):
            crop_box = expand_box(detection.box, self.crop_margin, width, height)
            crop = source.crop(crop_box)
            state_result = self.predict_state(crop)
            state_ms += state_result.get('timing_ms') or 0
            state = state_result['state']
            reason = state_result['reason']
            center_pct = None
            if state == 'unchecked' and state_result.get('state_box_px'):
                local_x, local_y = box_center(state_result['state_box_px'])
                center_x = local_x + crop_box[0]
                center_y = local_y + crop_box[1]
                if 0 <= center_x <= width and 0 <= center_y <= height:
                    center_pct = {'x': _round(center_x / width * 100),
                                  'y': _round(center_y / height * 100)}
                else:
                    state, reason = 'unknown', 'checkbox_center_outside_image'
            challenges.append({
                'state': state,
                'confidence': state_result['confidence'],
                'reason': reason,
                'checkbox_center_pct': center_pct,
                'widget_box_px': [_round(v) for v in detection.box],
                'widget_box_pct': box_to_pct(detection.box, width, height),
                'crop_px': list(crop_box),
            })
        total_ms = (time.perf_counter() - started) * 1000
        if len(challenges) == 1:
            challenge = challenges[0]
            summary = {'has_cf_challenge': True,
                       'state': challenge['state'],
                       'confidence': challenge['confidence'],
                       'checkbox_center_pct': challenge['checkbox_center_pct'],
                       'ambiguous': False,
                       'reason': challenge['reason']}
        else:
            summary = {'has_cf_challenge': True,
                       'state': 'unknown',
                       'confidence': None,
                       'checkbox_center_pct': None,
                       'ambiguous': True,
                       'reason': 'multiple_widgets'}
        return {**summary, 'challenges': challenges, 'image_size': [width, height],
                'timing_ms': {'widget': _round(widget_ms), 'state': _round(state_ms),
                              'total': _round(total_ms)}}

    def annotate(self, image: str | Path | Image.Image, result: dict[str, Any]) -> Image.Image:
        return annotate_result(self._as_image(image), result)
