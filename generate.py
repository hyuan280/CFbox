"""Generate synthetic UI detection datasets through the Camofox REST API."""

import argparse
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.parse import unquote, urlencode, quote, urlsplit
from urllib.request import Request, urlopen
import uuid

from PIL import Image

ROOT = Path(__file__).resolve().parent
STATES = ['unchecked', 'verified', 'processing', 'error', 'absent']
# Template families are disjoint across splits to reduce layout leakage.
FAMILIES = {'train': ['portal', 'article', 'dashboard', 'table', 'list', 'form', 'sidebar'],
            'val': ['checkout', 'grid'],
            'test': ['support', 'settings', 'dialog']}


class Camofox:
    def __init__(self, base, user):
        self.base = base.rstrip('/')
        self.user = user

    def request(self, method, path, data=None, binary=False):
        headers = {'Content-Type': 'application/json'}
        if os.environ.get('CAMOFOX_API_KEY'):
            headers['Authorization'] = 'Bearer ' + os.environ['CAMOFOX_API_KEY']
        request = Request(self.base + path,
                          data=json.dumps(data).encode() if data is not None else None,
                          headers=headers, method=method)
        try:
            with urlopen(request, timeout=120) as response:
                payload = response.read()
        except HTTPError as exc:
            raise RuntimeError(f'Camofox {method} {path}: {exc.code} '
                               f'{exc.read().decode(errors="replace")}') from exc
        return payload if binary else json.loads(payload)

    def evaluate(self, tab, expression):
        response = self.request('POST', f'/tabs/{quote(tab, safe="")}/evaluate',
                                {'userId': self.user, 'expression': expression})
        if 'result' not in response:
            raise RuntimeError(f'Unexpected evaluate response: {response}')
        return response['result']


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


@contextmanager
def serve(host, port, real_dir=None):
    real_root = Path(real_dir).resolve() if real_dir else None

    class Handler(QuietHandler):
        def translate_path(self, path):
            if real_root is not None:
                request_path = urlsplit(path).path
                prefix = '/real/'
                if request_path.startswith(prefix):
                    name = unquote(request_path[len(prefix):])
                    candidate = (real_root / name).resolve()
                    if candidate.parent == real_root and candidate.is_file():
                        return str(candidate)
                    return str(real_root / '__missing__')
            return super().translate_path(path)

    server = ThreadingHTTPServer((host, port), partial(Handler, directory=str(ROOT / 'web')))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


REAL_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def list_real_images(directory):
    root = Path(directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Real image directory not found: {root}')
    images = sorted(path for path in root.iterdir()
                    if path.is_file() and path.suffix.lower() in REAL_SUFFIXES)
    if not images:
        raise ValueError(f'No images found in {root}')
    return images


def split_real_images(images, seed):
    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    if count < 3:
        return {'train': shuffled, 'val': [], 'test': []}
    val_count = max(1, round(count * 0.1))
    test_count = max(1, round(count * 0.1))
    train_count = max(1, count - val_count - test_count)
    while train_count + val_count + test_count > count:
        if test_count > 1:
            test_count -= 1
        elif val_count > 1:
            val_count -= 1
        else:
            break
    return {'train': shuffled[:train_count],
            'val': shuffled[train_count:train_count + val_count],
            'test': shuffled[train_count + val_count:train_count + val_count + test_count]}


def real_viewport(path, max_size):
    with Image.open(path) as image:
        width, height = image.size
    if max_size and max(width, height) > max_size:
        scale = max_size / max(width, height)
        width = max(1, round(width * scale))
        height = max(1, round(height * scale))
    return width, height


def yolo_line(class_id, box, width, height):
    x1, y1, x2, y2 = box
    if not all(math.isfinite(v) for v in box) or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f'Invalid box {box} for {width}x{height}')
    return (f'{class_id} {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} '
            f'{(x2-x1)/width:.8f} {(y2-y1)/height:.8f}\n')


def save_sample(output, name, split, png, annotation):
    image = Image.open(io.BytesIO(png))
    image.load()
    width, height = image.size
    viewport = annotation['viewport']
    sx, sy = width / viewport['width'], height / viewport['height']
    def pixels(box):
        return [box[0]*sx, box[1]*sy, box[2]*sx, box[3]*sy] if box else None
    widget, checkbox = pixels(annotation['widget']), pixels(annotation['checkbox'])
    state = annotation['state']
    if (state == 'absent') != (widget is None) or (state == 'unchecked') != (checkbox is not None):
        raise ValueError('State and DOM annotations disagree')
    label = yolo_line(0, widget, width, height) if widget else ''
    center = None
    if checkbox:
        yolo_line(0, checkbox, width, height)
        center = {'x': (checkbox[0]+checkbox[2])/2/width*100,
                  'y': (checkbox[1]+checkbox[3])/2/height*100}
    metadata = {**annotation, 'source': annotation.get('source', 'synthetic'), 'split': split,
                'image_size': [width, height], 'widget_pixels': widget,
                'checkbox_pixels': checkbox, 'has_cf_challenge': widget is not None,
                'checkbox_center_pct': center}
    if widget:
        x1, y1, x2, y2 = widget
        crop = [max(0, math.floor(x1)-8), max(0, math.floor(y1)-8),
                min(width, math.ceil(x2)+8), min(height, math.ceil(y2)+8)]
        cw, ch = crop[2]-crop[0], crop[3]-crop[1]
        target = checkbox if checkbox else widget
        local = [target[0]-crop[0], target[1]-crop[1],
                 target[2]-crop[0], target[3]-crop[1]]
        crop_label = yolo_line(STATES.index(state), local, cw, ch)
        metadata['crop_pixels'] = crop
        image.crop(crop).save(output / 'state/images' / split / f'{name}.png')
        (output / 'state/labels' / split / f'{name}.txt').write_text(crop_label)
    (output / 'widget/images' / split / f'{name}.png').write_bytes(png)
    (output / 'widget/labels' / split / f'{name}.txt').write_text(label)
    (output / 'metadata' / f'{name}.json').write_text(json.dumps(metadata, indent=2))


def ensure_layout(output):
    for stage, names in [('widget', ['cf_widget']),
                         ('state', ['checkbox_unchecked', 'state_verified', 'state_processing', 'state_error'])]:
        for split in FAMILIES:
            for kind in ['images', 'labels']:
                (output / stage / kind / split).mkdir(parents=True, exist_ok=True)
        config_path = output / stage / 'data.yaml'
        if not config_path.exists():
            config = 'train: images/train\nval: images/val\ntest: images/test\nnames:\n'
            config += ''.join(f'  {i}: {name}\n' for i, name in enumerate(names))
            config_path.write_text(config)
    (output / 'metadata').mkdir(parents=True, exist_ok=True)


def initialize(output):
    # Refuse reuse so old files cannot silently leak between dataset splits.
    output.mkdir(parents=True, exist_ok=False)
    ensure_layout(output)


def sample_complete(output, name, split, state):
    if not (output / 'metadata' / f'{name}.json').is_file():
        return False
    if not (output / 'widget/images' / split / f'{name}.png').is_file():
        return False
    if not (output / 'widget/labels' / split / f'{name}.txt').is_file():
        return False
    if state != 'absent':
        if not (output / 'state/images' / split / f'{name}.png').is_file():
            return False
        if not (output / 'state/labels' / split / f'{name}.txt').is_file():
            return False
    return True


def generate(args):
    output = Path(args.output).resolve()
    run_path = output / 'run.json'
    client = Camofox(args.api, 'cfbox-' + uuid.uuid4().hex[:12])
    client.request('GET', '/health')

    if args.resume:
        if not run_path.is_file():
            raise SystemExit(f'error: cannot resume, missing {run_path}')
        run = json.loads(run_path.read_text())
        plan = run.get('plan')
        if not isinstance(plan, list) or not plan:
            raise SystemExit('error: run.json has no plan to resume')
        if run.get('status') == 'complete':
            print(f'Dataset already complete: {output}')
            return
        real_items = [item['real'] for item in plan if item.get('real')]
        if real_items:
            if not args.real_images:
                raise SystemExit('error: --real-images is required to resume this dataset')
            available = {path.name for path in list_real_images(args.real_images)}
            missing = sorted(set(real_items) - available)
            if missing:
                raise SystemExit(f'error: missing real backgrounds: {", ".join(missing)}')
        ensure_layout(output)
        run['status'] = 'incomplete'
        run_path.write_text(json.dumps(run, indent=2))
        print(f'Resuming {len(plan)} samples from {output}', flush=True)
    else:
        if output.exists():
            raise SystemExit(f'error: output exists: {output}; use --resume to continue it')
        initialize(output)
        rng = random.Random(args.seed)
        real_by_split = {'train': [], 'val': [], 'test': []}
        if args.real_images:
            real_by_split = split_real_images(list_real_images(args.real_images), args.seed)
        real_sizes = {}
        plan = []
        for index in range(args.count):
            # Round-robin gives every split samples even in small smoke runs.
            split = ['val', 'test', 'train', 'train', 'train', 'train', 'train', 'train', 'train', 'train'][index % 10]
            state = STATES[(index // 10 + index % 10) % 5]
            use_real = bool(real_by_split[split]) and rng.random() < args.real_ratio
            if use_real:
                real_path = rng.choice(real_by_split[split])
                if real_path not in real_sizes:
                    real_sizes[real_path] = real_viewport(real_path, args.real_max_size)
                plan.append({'index': index, 'split': split, 'state': state,
                             'family': 'real', 'seed': rng.randrange(1, 2**32),
                             'viewport': real_sizes[real_path], 'real': real_path.name})
            else:
                plan.append({'index': index, 'split': split, 'state': state,
                             'family': rng.choice(FAMILIES[split]), 'seed': rng.randrange(1, 2**32),
                             'viewport': rng.choice([(1280, 720), (1440, 900), (1920, 1080), (800, 800)])})
        run = {'seed': args.seed, 'count': args.count, 'plan': plan, 'status': 'incomplete'}
        run_path.write_text(json.dumps(run, indent=2))

    tabs = {}
    user_prefix = client.user
    users = set()
    generated = 0
    skipped = 0
    try:
        with serve(args.host, args.port, args.real_images) as port:
            base = (args.page_base or f'http://127.0.0.1:{port}').rstrip('/')
            for item in plan:
                name = f'sample_{item["index"]:06d}'
                split = item['split']
                if sample_complete(output, name, split, item['state']):
                    skipped += 1
                    continue
                viewport = tuple(item['viewport'])
                client.user = f'{user_prefix}-{viewport[0]}x{viewport[1]}'
                users.add(client.user)
                query_data = {k: item[k] for k in ['seed', 'state', 'family']}
                if item.get('real'):
                    query_data['real'] = item['real']
                url = base + '/?' + urlencode(query_data)
                last_error = None
                for attempt in range(1, args.retries + 1):
                    try:
                        if viewport not in tabs:
                            response = client.request('POST', '/tabs', {'userId': client.user,
                                'sessionKey': 'dataset', 'url': url,
                                'viewport': {'width': viewport[0], 'height': viewport[1]}})
                            tabs[viewport] = response['tabId']
                        else:
                            client.request('POST', f'/tabs/{tabs[viewport]}/navigate',
                                           {'userId': client.user, 'url': url})
                        tab = tabs[viewport]
                        deadline = time.monotonic() + 30
                        while not client.evaluate(tab, "document.body.dataset.ready === 'true'"):
                            if time.monotonic() > deadline:
                                raise TimeoutError('Synthetic page did not become ready')
                            time.sleep(.1)
                        annotation = client.evaluate(tab, """(() => {
                            const rect = id => {
                                const el = document.getElementById(id);
                                if (!el) return null;
                                const r = el.getBoundingClientRect();
                                return [r.x, r.y, r.right, r.bottom];
                            };
                            return {state: document.body.dataset.state,
                                family: document.body.dataset.family,
                                viewport: {width: innerWidth, height: innerHeight},
                                widget: rect('cf-widget'), checkbox: rect('checkbox')};
                        })()""")
                        if annotation['state'] != item['state'] or annotation['family'] != item['family']:
                            raise RuntimeError('Browser page does not match requested sample')
                        png = client.request('GET', f'/tabs/{tab}/screenshot?' + urlencode(
                            {'userId': client.user, 'fullPage': 'false'}), binary=True)
                        extra = {'seed': item['seed']}
                        if item.get('real'):
                            extra.update(source='real-background', background_image=item['real'])
                        save_sample(output, name, split, png, {**annotation, **extra})
                        print(f'{name}: {split} {item["state"]}', flush=True)
                        generated += 1
                        break
                    except Exception as exc:
                        last_error = exc
                        print(f'warning: {name} failed on attempt '
                              f'{attempt}/{args.retries}: {exc}', file=sys.stderr, flush=True)
                        tabs.pop(viewport, None)
                        try:
                            client.request('DELETE', '/sessions/' + client.user)
                        except Exception:
                            pass
                        if attempt < args.retries:
                            time.sleep(args.retry_delay * attempt)
                else:
                    raise RuntimeError(f'{name} failed after {args.retries} attempts: {last_error}')
    finally:
        for user in users:
            try:
                client.request('DELETE', '/sessions/' + user)
            except Exception:
                pass
    run = json.loads(run_path.read_text())
    run['status'] = 'complete'
    run_path.write_text(json.dumps(run, indent=2))
    print(f'Dataset complete: {output} (generated {generated}, skipped {skipped})')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='datasets/synthetic-v1')
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--api', default=os.environ.get('CAMOFOX_URL', 'http://localhost:9377'))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--page-base', help='Browser-reachable URL of this web server (for remote/container Camofox)')
    parser.add_argument('--real-images', type=Path,
                        help='Directory of real screenshots without a CF widget to use as backgrounds')
    parser.add_argument('--real-ratio', type=float, default=1.0,
                        help='Fraction of samples using real backgrounds, 0..1 (default: 1.0)')
    parser.add_argument('--real-max-size', type=int, default=1920,
                        help='Scale real backgrounds to this maximum side, 0 disables (default: 1920)')
    parser.add_argument('--resume', action='store_true',
                        help='Continue an interrupted dataset in --output')
    parser.add_argument('--retries', type=int, default=5,
                        help='Attempts per sample before stopping (default: 5)')
    parser.add_argument('--retry-delay', type=float, default=2.0,
                        help='Initial retry delay in seconds (default: 2.0)')
    parser.add_argument('--serve-only', action='store_true', help='Preview the synthetic page without capturing')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    if not 0 <= args.real_ratio <= 1:
        parser.error('--real-ratio must be between 0 and 1')
    if args.real_max_size < 0:
        parser.error('--real-max-size must be non-negative')
    if args.retries < 1:
        parser.error('--retries must be positive')
    if args.retry_delay < 0:
        parser.error('--retry-delay must be non-negative')
    if args.serve_only:
        real_images = list_real_images(args.real_images) if args.real_images else []
        with serve(args.host, args.port, args.real_images) as port:
            preview = f'http://{args.host}:{port}/?seed=42&state=unchecked&family=portal'
            if real_images:
                preview += '&real=' + quote(real_images[0].name)
            print(f'Preview: {preview}', flush=True)
            try:
                threading.Event().wait()
            except KeyboardInterrupt:
                pass
    else:
        generate(args)


if __name__ == '__main__':
    main()
