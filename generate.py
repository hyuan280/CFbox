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
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen
import uuid

from PIL import Image

ROOT = Path(__file__).resolve().parent
STATES = ['unchecked', 'verified', 'processing', 'error', 'absent']
# Template families are disjoint across splits to reduce layout leakage.
FAMILIES = {'train': ['portal', 'article', 'dashboard'], 'val': ['checkout'],
            'test': ['support', 'settings']}


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
def serve(host, port):
    server = ThreadingHTTPServer((host, port), partial(QuietHandler, directory=str(ROOT / 'web')))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


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
    metadata = {**annotation, 'source': 'synthetic', 'split': split,
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


def initialize(output):
    # Refuse reuse so old files cannot silently leak between dataset splits.
    output.mkdir(parents=True, exist_ok=False)
    for stage, names in [('widget', ['cf_widget']),
                         ('state', ['checkbox_unchecked', 'state_verified', 'state_processing', 'state_error'])]:
        for split in FAMILIES:
            for kind in ['images', 'labels']:
                (output / stage / kind / split).mkdir(parents=True)
        config = 'train: images/train\nval: images/val\ntest: images/test\nnames:\n'
        config += ''.join(f'  {i}: {name}\n' for i, name in enumerate(names))
        (output / stage / 'data.yaml').write_text(config)
    (output / 'metadata').mkdir()


def generate(args):
    output = Path(args.output).resolve()
    client = Camofox(args.api, 'cfbox-' + uuid.uuid4().hex[:12])
    client.request('GET', '/health')
    initialize(output)
    rng = random.Random(args.seed)
    plan = []
    for index in range(args.count):
        # Round-robin gives every split samples even in small smoke runs.
        split = ['val', 'test', 'train', 'train', 'train', 'train', 'train', 'train', 'train', 'train'][index % 10]
        plan.append({'index': index, 'split': split, 'state': STATES[(index // 10 + index % 10) % 5],
                     'family': rng.choice(FAMILIES[split]), 'seed': rng.randrange(1, 2**32),
                     'viewport': rng.choice([(1280, 720), (1440, 900), (1920, 1080), (800, 800)])})
    (output / 'run.json').write_text(json.dumps({'seed': args.seed, 'count': args.count,
                                               'plan': plan, 'status': 'incomplete'}, indent=2))
    tabs = {}
    user_prefix = client.user
    users = set()
    try:
        with serve(args.host, args.port) as port:
            base = (args.page_base or f'http://127.0.0.1:{port}').rstrip('/')
            for item in plan:
                viewport = item['viewport']
                client.user = f'{user_prefix}-{viewport[0]}x{viewport[1]}'
                users.add(client.user)
                query = urlencode({k: item[k] for k in ['seed', 'state', 'family']})
                url = base + '/?' + query
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
                name = f'sample_{item["index"]:06d}'
                save_sample(output, name, item['split'], png, {**annotation, 'seed': item['seed']})
                print(f'{name}: {item["split"]} {item["state"]}', flush=True)
    finally:
        # This user is unique to the run; never close another user's tabs.
        for user in users:
            client.request('DELETE', '/sessions/' + user)
    run_path = output / 'run.json'
    run = json.loads(run_path.read_text())
    run['status'] = 'complete'
    run_path.write_text(json.dumps(run, indent=2))
    print(f'Dataset saved to {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='datasets/synthetic-v1')
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--api', default=os.environ.get('CAMOFOX_URL', 'http://localhost:9377'))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--page-base', help='Browser-reachable URL of this web server (for remote/container Camofox)')
    parser.add_argument('--serve-only', action='store_true', help='Preview the synthetic page without capturing')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    if args.serve_only:
        with serve(args.host, args.port) as port:
            print(f'Preview: http://{args.host}:{port}/?seed=42&state=unchecked&family=portal', flush=True)
            try:
                threading.Event().wait()
            except KeyboardInterrupt:
                pass
    else:
        generate(args)


if __name__ == '__main__':
    main()
