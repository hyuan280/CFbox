"""Regenerate Chinese generated samples with the bundled CJK font."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import time
import uuid
from urllib.parse import urlencode

from generate import Camofox, list_real_images, save_sample, serve

ARTICLE_FAMILIES = {'article', 'dashboard', 'support', 'real'}
NO_RANDOM_FAMILIES = {'portal', 'checkout', 'settings', 'table', 'list',
                      'form', 'dialog', 'sidebar', 'grid'}


def language_for(seed: int, family: str) -> str:
    calls = 9 if family in ARTICLE_FAMILIES or family not in NO_RANDOM_FAMILIES else 3
    value = 0
    state = int(seed) & 0xffffffff
    for _ in range(calls + 1):
        state = (state * 1664525 + 1013904223) & 0xffffffff
        value = state / 4294967296
    return 'en' if value < 0.5 else 'zh'


def load_dataset(dataset: Path):
    run_path = dataset / 'run.json'
    if not run_path.is_file():
        raise RuntimeError(f'missing run.json: {run_path}')
    run = json.loads(run_path.read_text())
    plan = run.get('plan')
    if not isinstance(plan, list) or not plan:
        raise RuntimeError(f'run.json has no plan: {run_path}')
    return plan


def select_items(dataset: Path, plan):
    selected = []
    counts = {'selected': 0, 'already_fixed': 0, 'english': 0, 'absent': 0, 'missing': 0}
    for item in plan:
        name = f'sample_{item["index"]:06d}'
        meta_path = dataset / 'metadata' / f'{name}.json'
        if not meta_path.is_file():
            counts['missing'] += 1
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get('font_fixed') or meta.get('language') in {'en', 'zh'}:
            counts['already_fixed'] += 1
            continue
        if item.get('state') == 'absent':
            counts['absent'] += 1
            continue
        family = item.get('family', meta.get('family', ''))
        if language_for(item['seed'], family) != 'zh':
            counts['english'] += 1
            continue
        selected.append((item, name, meta))
        counts['selected'] += 1
    return selected, counts


def capture(item, name, dataset, client, tabs, users, base, args):
    viewport = tuple(item['viewport'])
    client.user = f'{args.user_prefix}-{viewport[0]}x{viewport[1]}'
    users.add(client.user)
    query = {key: item[key] for key in ['seed', 'state', 'family']}
    query['lang'] = 'zh'
    if item.get('theme'):
        query['theme'] = item['theme']
    if item.get('real'):
        query['real'] = item['real']
    url = base + '/?' + urlencode(query)
    last_error = None
    for attempt in range(1, args.retries + 1):
        try:
            if viewport not in tabs:
                response = client.request('POST', '/tabs', {
                    'userId': client.user, 'sessionKey': 'repair', 'url': url,
                    'viewport': {'width': viewport[0], 'height': viewport[1]}})
                tabs[viewport] = response['tabId']
            else:
                client.request('POST', f'/tabs/{tabs[viewport]}/navigate',
                               {'userId': client.user, 'url': url})
            tab = tabs[viewport]
            deadline = time.monotonic() + 30
            while not client.evaluate(tab, "document.body.dataset.ready === 'true'"):
                if time.monotonic() > deadline:
                    raise TimeoutError('page did not become ready')
                time.sleep(.1)
            annotation = client.evaluate(tab, """(() => {
                const rect = id => {
                    const el = document.getElementById(id);
                    if (!el) return null;
                    const r = el.getBoundingClientRect();
                    return [r.x, r.y, r.right, r.bottom];
                };
                return {
                    state: document.body.dataset.state,
                    family: document.body.dataset.family,
                    theme: document.body.dataset.theme,
                    language: document.body.dataset.language,
                    viewport: {width: innerWidth, height: innerHeight},
                    widget: rect('cf-widget'),
                    checkbox: rect('checkbox')
                };
            })()""")
            if annotation['state'] != item['state'] or annotation['family'] != item['family']:
                raise RuntimeError('page does not match plan')
            png = client.request('GET',
                                 f'/tabs/{tab}/screenshot?' + urlencode(
                                     {'userId': client.user, 'fullPage': 'false'}),
                                 binary=True)
            extra = {'seed': item['seed'], 'language': 'zh', 'font_fixed': True}
            if item.get('real'):
                extra.update(source='real-background', background_image=item['real'])
            save_sample(dataset, name, item['split'], png, {**annotation, **extra})
            return True, None
        except Exception as exc:
            last_error = exc
            print(f'warning: {dataset.name}/{name} attempt {attempt}/'
                  f'{args.retries} failed: {exc}', file=sys.stderr, flush=True)
            tabs.pop(viewport, None)
            try:
                client.request('DELETE', '/sessions/' + client.user)
            except Exception:
                pass
            if attempt < args.retries:
                time.sleep(args.retry_delay * attempt)
    return False, last_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, nargs='+', required=True)
    parser.add_argument('--real-images', type=Path)
    parser.add_argument('--api', default='http://127.0.0.1:9377')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--retries', type=int, default=5)
    parser.add_argument('--retry-delay', type=float, default=2.0)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    args.user_prefix = 'cfbox-repair-' + uuid.uuid4().hex[:10]

    all_selected = {}
    for dataset in args.dataset:
        plan = load_dataset(dataset)
        selected, counts = select_items(dataset, plan)
        all_selected[dataset] = (plan, selected)
        print(f'{dataset}: selected={counts["selected"]} '
              f'english={counts["english"]} absent={counts["absent"]} '
              f'already_fixed={counts["already_fixed"]} missing={counts["missing"]}')
    total = sum(len(selected) for _, selected in all_selected.values())
    if args.dry_run:
        print(f'dry run: {total} Chinese non-absent samples would be regenerated')
        return
    if total == 0:
        print('nothing to regenerate')
        return
    if any(item.get('real') for _, selected in all_selected.values()
           for item, _, _ in selected):
        if not args.real_images:
            sys.exit('error: --real-images is required for real-background samples')
        list_real_images(args.real_images)

    client = Camofox(args.api, args.user_prefix)
    client.request('GET', '/health')
    tabs = {}
    users = set()
    repaired = failed = 0
    try:
        with serve(args.host, args.port, args.real_images) as port:
            base = f'http://127.0.0.1:{port}'
            for dataset, (_, selected) in all_selected.items():
                for item, name, _ in selected:
                    ok, error = capture(item, name, dataset, client, tabs, users,
                                        base, args)
                    if ok:
                        repaired += 1
                        print(f'repaired {dataset.name}/{name}', flush=True)
                    else:
                        failed += 1
                        print(f'FAILED {dataset.name}/{name}: {error}', file=sys.stderr)
    finally:
        for user in users:
            try:
                client.request('DELETE', '/sessions/' + user)
            except Exception:
                pass
    print(f'repaired={repaired} failed={failed}')
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
