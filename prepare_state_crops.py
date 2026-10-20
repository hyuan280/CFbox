"""Validate real-screenshot metadata and build missing state crop images."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image

STATES = {'absent', 'unchecked', 'verified', 'processing', 'error'}
SPLITS = {'train', 'val', 'test'}
IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')


def find_widget_image(widget_dir: Path, name: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        path = widget_dir / f'{name}{suffix}'
        if path.is_file():
            return path
    return None


def valid_box(box, width: int, height: int) -> list[float] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        values = [float(value) for value in box]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        return None
    return values


def crop_box(widget: list[float], width: int, height: int,
             margin: int) -> list[int]:
    x1, y1, x2, y2 = widget
    return [max(0, math.floor(x1) - margin),
            max(0, math.floor(y1) - margin),
            min(width, math.ceil(x2) + margin),
            min(height, math.ceil(y2) + margin)]


def process(meta_path: Path, dataset: Path, args) -> tuple[str, str | None]:
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return 'error', f'{meta_path.name}: invalid JSON: {exc}'

    name = meta_path.stem
    split = meta.get('split')
    state = meta.get('state')
    if split not in SPLITS:
        return 'error', f'{name}: invalid or missing split'
    if state not in STATES:
        return 'error', f'{name}: invalid or missing state'
    has_widget = bool(meta.get('has_cf_challenge'))
    widget = meta.get('widget_pixels')
    checkbox = meta.get('checkbox_pixels')

    widget_dir = dataset / 'widget/images' / split
    widget_path = find_widget_image(widget_dir, name)
    if widget_path is None:
        return 'error', f'{name}: missing widget image in {widget_dir}'
    try:
        image = Image.open(widget_path).convert('RGB')
        image.load()
    except Exception as exc:
        return 'error', f'{name}: cannot open {widget_path}: {exc}'
    width, height = image.size
    png_path = widget_dir / f'{name}.png'
    if widget_path.suffix.lower() != '.png':
        if not args.dry_run:
            image.save(png_path)
        widget_path = png_path

    state_path = dataset / 'state/images' / split / f'{name}.png'

    if state == 'absent':
        if has_widget or widget is not None or checkbox is not None:
            return 'error', f'{name}: absent must have no widget/checkbox'
        if state_path.exists():
            if args.prune_stale and not args.dry_run:
                state_path.unlink()
                return 'pruned', None
            return 'warning', f'{name}: absent but stale state image exists'
        return 'skipped', None

    if not has_widget:
        return 'error', f'{name}: positive sample must set has_cf_challenge'
    if widget is None:
        return 'error', f'{name}: missing widget_pixels'
    widget_box = valid_box(widget, width, height)
    if widget_box is None:
        return 'error', f'{name}: widget_pixels outside image or invalid'

    if state == 'unchecked':
        if checkbox is None:
            return 'error', f'{name}: unchecked sample needs checkbox_pixels'
        checkbox_box = valid_box(checkbox, width, height)
        if checkbox_box is None:
            return 'error', f'{name}: checkbox_pixels outside image or invalid'
        if checkbox_box[0] < widget_box[0] or checkbox_box[1] < widget_box[1] \
                or checkbox_box[2] > widget_box[2] or checkbox_box[3] > widget_box[3]:
            return 'error', f'{name}: checkbox_pixels is not inside widget_pixels'
    elif checkbox is not None:
        return 'error', f'{name}: only unchecked samples may have checkbox_pixels'

    crop = crop_box(widget_box, width, height, args.margin)
    changed = False
    if meta.get('crop_pixels') != crop:
        meta['crop_pixels'] = crop
        changed = True
        print(f'{name}: crop_pixels set to {crop}')
    if meta.get('image_size') != [width, height]:
        meta['image_size'] = [width, height]
        changed = True
        print(f'{name}: image_size set to {[width, height]}')
    if changed and not args.dry_run:
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n')

    expected_size = (crop[2] - crop[0], crop[3] - crop[1])
    need = args.force or changed or not state_path.exists()
    if state_path.exists():
        try:
            with Image.open(state_path) as current:
                if current.size != expected_size:
                    need = True
        except Exception:
            need = True
        if state_path.stat().st_mtime < widget_path.stat().st_mtime:
            need = True
        if not changed and state_path.stat().st_mtime < meta_path.stat().st_mtime:
            need = True

    if not need:
        return 'skipped', None
    if args.dry_run:
        print(f'{name}: would generate {state_path.relative_to(dataset)} {expected_size}')
        return 'would-generate', None

    state_path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(tuple(crop)).save(state_path)
    print(f'{name}: generated {state_path.relative_to(dataset)} {expected_size}')
    return 'generated', None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('datasets/real'))
    parser.add_argument('--margin', type=int, default=8)
    parser.add_argument('--force', action='store_true',
                        help='Regenerate every positive state crop')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report actions without writing files')
    parser.add_argument('--prune-stale', action='store_true',
                        help='Remove state crops for absent samples')
    args = parser.parse_args()
    if args.margin < 0:
        parser.error('--margin must be non-negative')
    metadata_dir = args.dataset / 'metadata'
    if not metadata_dir.is_dir():
        sys.exit(f'error: metadata directory not found: {metadata_dir}')

    actions: dict[str, int] = {}
    errors = 0
    for meta_path in sorted(metadata_dir.glob('*.json')):
        action, message = process(meta_path, args.dataset, args)
        actions[action] = actions.get(action, 0) + 1
        if action == 'error':
            errors += 1
            print(f'ERROR {message}', file=sys.stderr)
        elif message:
            print(f'WARNING {message}', file=sys.stderr)

    print('summary:', ' '.join(f'{key}={value}' for key, value in sorted(actions.items())))
    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()
