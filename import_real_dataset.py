"""Build a train/val/test dataset from real screenshots and generated batches."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import re
import shutil
import sys

from PIL import Image

STATE_CLASS = {'unchecked': 0, 'verified': 1, 'processing': 2, 'error': 3}
SPLITS = ('train', 'val', 'test')
IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')


def find_image(directory: Path, name: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        path = directory / f'{name}{suffix}'
        if path.is_file():
            return path
    return None


def allocate(count: int, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if count <= 0:
        return 0, 0, 0
    if count == 1:
        return 1, 0, 0
    test_count = min(count - 1, max(1, round(count * test_ratio)))
    remaining = count - test_count
    val_count = 0
    if remaining >= 2:
        val_count = min(remaining - 1, max(1, round(count * val_ratio)))
    train_count = count - test_count - val_count
    return train_count, val_count, test_count


def yolo_line(class_id: int, box, width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    return (f'{class_id} {(x1 + x2) / 2 / width:.8f} {(y1 + y2) / 2 / height:.8f} '
            f'{(x2 - x1) / width:.8f} {(y2 - y1) / height:.8f}\n')


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
             margin: int = 8) -> list[int]:
    x1, y1, x2, y2 = widget
    return [max(0, math.floor(x1) - margin),
            max(0, math.floor(y1) - margin),
            min(width, math.ceil(x2) + margin),
            min(height, math.ceil(y2) + margin)]


def init_output(root: Path, force: bool):
    if root.exists():
        if not force:
            sys.exit(f'error: output already exists: {root}')
        shutil.rmtree(root)
    for stage, names in [
        ('widget', ['cf_widget']),
        ('state', ['checkbox_unchecked', 'state_verified',
                   'state_processing', 'state_error']),
    ]:
        for split in SPLITS:
            (root / stage / 'images' / split).mkdir(parents=True)
            (root / stage / 'labels' / split).mkdir(parents=True)
        config = 'train: images/train\nval: images/val\ntest: images/test\nnames:\n'
        config += ''.join(f'  {index}: {name}\n' for index, name in enumerate(names))
        (root / stage / 'data.yaml').write_text(config)
    (root / 'metadata').mkdir()


def assign_hascf(paths: list[Path], seed: int, val_ratio: float,
                 test_ratio: float) -> dict[str, str]:
    groups = defaultdict(list)
    for path in paths:
        meta = json.loads(path.read_text())
        groups[meta.get('state')].append(path)
    assignment = {}
    for state, state_paths in groups.items():
        shuffled = list(state_paths)
        random.Random(f'{seed}:{state}').shuffle(shuffled)
        train_count, val_count, test_count = allocate(len(shuffled), val_ratio, test_ratio)
        for index, path in enumerate(shuffled):
            if index < train_count:
                assignment[path.stem] = 'train'
            elif index < train_count + val_count:
                assignment[path.stem] = 'val'
            else:
                assignment[path.stem] = 'test'
    return assignment


def assign_plain(paths: list[Path], seed: int, val_ratio: float,
                 test_ratio: float) -> dict[str, str]:
    shuffled = [path.stem for path in paths]
    random.Random(seed).shuffle(shuffled)
    train_count, val_count, _ = allocate(len(shuffled), val_ratio, test_ratio)
    assignment = {}
    for index, name in enumerate(shuffled):
        if index < train_count:
            assignment[name] = 'train'
        elif index < train_count + val_count:
            assignment[name] = 'val'
        else:
            assignment[name] = 'test'
    return assignment


def validate_hascf(src: Path, errors: list[str]):
    for meta_path in sorted(src.glob('*.json')):
        name = meta_path.stem
        try:
            raw = json.loads(meta_path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f'{name}: invalid JSON: {exc}')
            continue
        state = raw.get('state')
        if state not in STATE_CLASS:
            errors.append(f'{name}: unsupported state {state!r}')
            continue
        image_path = find_image(src, name)
        if image_path is None:
            errors.append(f'{name}: image missing')
            continue
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except Exception as exc:
            errors.append(f'{name}: cannot open image: {exc}')
            continue
        if valid_box(raw.get('widget_pixels'), width, height) is None:
            errors.append(f'{name}: widget_pixels is invalid or outside the image')
        checkbox = raw.get('checkbox_pixels')
        checkbox_box = None
        if checkbox is not None:
            checkbox_box = valid_box(checkbox, width, height)
            if checkbox_box is None:
                errors.append(f'{name}: checkbox_pixels is invalid or outside the image')
        if state == 'unchecked' and checkbox_box is None:
            errors.append(f'{name}: unchecked sample needs checkbox_pixels')
        if state != 'unchecked' and checkbox_box is not None:
            errors.append(f'{name}: only unchecked may have checkbox_pixels')


def validate_nocf(src: Path, errors: list[str]):
    for image_path in sorted(path for path in src.iterdir()
                             if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES):
        try:
            with Image.open(image_path) as image:
                image.load()
        except Exception as exc:
            errors.append(f'{image_path.stem}: cannot open image: {exc}')


def validate_extra(src: Path, errors: list[str]):
    run_path = src / 'run.json'
    if run_path.is_file():
        try:
            run = json.loads(run_path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f'{src.name}: invalid run.json: {exc}')
            return
        if run.get('status') != 'complete':
            errors.append(f'{src.name}: generation status is not complete')
            return
    metadata_dir = src / 'metadata'
    if not metadata_dir.is_dir():
        errors.append(f'{src.name}: metadata directory missing')
        return
    for meta_path in sorted(metadata_dir.glob('*.json')):
        name = meta_path.stem
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f'{src.name}/{name}: invalid JSON: {exc}')
            continue
        split = meta.get('split')
        if split not in SPLITS:
            errors.append(f'{src.name}/{name}: invalid split')
            continue
        if find_image(src / 'widget/images' / split, name) is None:
            errors.append(f'{src.name}/{name}: widget image missing')
        if meta.get('has_cf_challenge'):
            if not (src / 'state/images' / split / f'{name}.png').is_file():
                errors.append(f'{src.name}/{name}: state crop missing')
            if not (src / 'state/labels' / split / f'{name}.txt').is_file():
                errors.append(f'{src.name}/{name}: state label missing')


def import_hascf(src: Path, root: Path, assignment: dict[str, str],
                 counts: Counter, errors: list[str]):
    images_dir = root / 'widget/images'
    labels_dir = root / 'widget/labels'
    state_images = root / 'state/images'
    state_labels = root / 'state/labels'
    for meta_path in sorted(src.glob('*.json')):
        name = meta_path.stem
        split = assignment.get(name)
        if split is None:
            errors.append(f'{name}: no split assigned')
            continue
        image_path = find_image(src, name)
        if image_path is None:
            errors.append(f'{name}: image missing')
            continue
        try:
            raw = json.loads(meta_path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f'{name}: invalid JSON: {exc}')
            continue
        state = raw.get('state')
        if state not in STATE_CLASS:
            errors.append(f'{name}: unsupported state {state!r}')
            continue
        image = Image.open(image_path).convert('RGB')
        image.load()
        width, height = image.size
        widget = valid_box(raw.get('widget_pixels'), width, height)
        if widget is None:
            errors.append(f'{name}: invalid widget_pixels')
            continue
        checkbox = raw.get('checkbox_pixels')
        checkbox_box = None
        if checkbox is not None:
            checkbox_box = valid_box(checkbox, width, height)
            if checkbox_box is None:
                errors.append(f'{name}: invalid checkbox_pixels')
                continue
        if state == 'unchecked' and checkbox_box is None:
            errors.append(f'{name}: unchecked sample needs checkbox_pixels')
            continue
        if state != 'unchecked' and checkbox_box is not None:
            errors.append(f'{name}: only unchecked may have checkbox_pixels')
            continue
        crop = crop_box(widget, width, height)
        cw, ch = crop[2] - crop[0], crop[3] - crop[1]
        target = checkbox_box if state == 'unchecked' else widget
        local = [target[0] - crop[0], target[1] - crop[1],
                 target[2] - crop[0], target[3] - crop[1]]
        image.save(images_dir / split / f'{name}.png')
        (labels_dir / split / f'{name}.txt').write_text(
            yolo_line(0, widget, width, height))
        image.crop(tuple(crop)).save(state_images / split / f'{name}.png')
        (state_labels / split / f'{name}.txt').write_text(
            yolo_line(STATE_CLASS[state], local, cw, ch))
        metadata = {
            'split': split,
            'state': state,
            'has_cf_challenge': True,
            'widget_pixels': widget,
            'checkbox_pixels': checkbox_box,
            'crop_pixels': crop,
            'image_size': [width, height],
            'source': 'real-hascf',
        }
        (root / 'metadata' / f'{name}.json').write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
        counts[('hascf', split)] += 1


def import_nocf(src: Path, root: Path, assignment: dict[str, str],
                counts: Counter, errors: list[str]):
    images_dir = root / 'widget/images'
    labels_dir = root / 'widget/labels'
    for image_path in sorted(path for path in src.iterdir()
                             if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES):
        name = image_path.stem
        split = assignment.get(name)
        if split is None:
            errors.append(f'{name}: no split assigned')
            continue
        image = Image.open(image_path).convert('RGB')
        image.load()
        width, height = image.size
        image.save(images_dir / split / f'{name}.png')
        (labels_dir / split / f'{name}.txt').write_text('')
        metadata = {
            'split': split,
            'state': 'absent',
            'has_cf_challenge': False,
            'widget_pixels': None,
            'checkbox_pixels': None,
            'image_size': [width, height],
            'source': 'real-nocf',
        }
        (root / 'metadata' / f'{name}.json').write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
        counts[('nocf', split)] += 1


def import_extra(src: Path, root: Path, counts: Counter, errors: list[str]):
    run_path = src / 'run.json'
    if run_path.is_file():
        run = json.loads(run_path.read_text())
        if run.get('status') != 'complete':
            errors.append(f'{src}: generation status is not complete')
            return
    prefix = re.sub(r'[^0-9A-Za-z]+', '_', src.name).strip('_') + '_'
    for meta_path in sorted((src / 'metadata').glob('*.json')):
        name = meta_path.stem
        new_name = prefix + name
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f'{src.name}/{name}: invalid JSON: {exc}')
            continue
        split = meta.get('split')
        if split not in SPLITS:
            errors.append(f'{src.name}/{name}: invalid split')
            continue
        image_path = find_image(src / 'widget/images' / split, name)
        if image_path is None:
            errors.append(f'{src.name}/{name}: widget image missing')
            continue
        shutil.copy2(image_path,
                     root / 'widget/images' / split / f'{new_name}.png')
        widget_label = src / 'widget/labels' / split / f'{name}.txt'
        (root / 'widget/labels' / split / f'{new_name}.txt').write_text(
            widget_label.read_text() if widget_label.is_file() else '')
        if meta.get('has_cf_challenge'):
            state_path = src / 'state/images' / split / f'{name}.png'
            state_label = src / 'state/labels' / split / f'{name}.txt'
            if not state_path.is_file() or not state_label.is_file():
                errors.append(f'{src.name}/{name}: state crop or label missing')
                continue
            shutil.copy2(state_path,
                         root / 'state/images' / split / f'{new_name}.png')
            shutil.copy2(state_label,
                         root / 'state/labels' / split / f'{new_name}.txt')
        (root / 'metadata' / f'{new_name}.json').write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + '\n')
        counts[(src.name, split)] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('datasets/real-v1'))
    parser.add_argument('--hascf', type=Path, default=Path('images/hascf'))
    parser.add_argument('--nocf', type=Path, default=Path('images/nocf'))
    parser.add_argument('--extra', type=Path, nargs='*', default=[],
                        help='Generated dataset directories to merge')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.2)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.val_ratio < 1 or not 0 < args.test_ratio < 1:
        parser.error('split ratios must be between 0 and 1')
    if args.val_ratio + args.test_ratio >= 1:
        parser.error('val-ratio + test-ratio must be less than 1')

    hascf_paths = sorted(args.hascf.glob('*.json'))
    nocf_paths = sorted(path for path in args.nocf.iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not hascf_paths:
        parser.error(f'no hascf annotations found in {args.hascf}')
    if not nocf_paths:
        parser.error(f'no nocf images found in {args.nocf}')

    errors: list[str] = []
    validate_hascf(args.hascf, errors)
    validate_nocf(args.nocf, errors)
    prefixes = {}
    for extra in args.extra:
        if not (extra / 'metadata').is_dir():
            errors.append(f'{extra}: not a generated dataset')
            continue
        prefix = re.sub(r'[^0-9A-Za-z]+', '_', extra.name).strip('_') + '_'
        if prefix in prefixes:
            errors.append(f'{extra}: output name prefix collides with {prefixes[prefix]}')
        prefixes[prefix] = extra
        validate_extra(extra, errors)
    if errors:
        for error in errors:
            print(f'ERROR {error}', file=sys.stderr)
        sys.exit(1)
    if args.output.exists() and not args.force:
        sys.exit(f'error: output exists: {args.output}; use --force to replace it')

    hascf_split = assign_hascf(hascf_paths, args.seed,
                               args.val_ratio, args.test_ratio)
    nocf_split = assign_plain(nocf_paths, args.seed,
                              args.val_ratio, args.test_ratio)

    temp = args.output.parent / f'.{args.output.name}.tmp'
    if temp.exists():
        shutil.rmtree(temp)
    init_output(temp, False)
    counts = Counter()
    try:
        import_hascf(args.hascf, temp, hascf_split, counts, errors)
        import_nocf(args.nocf, temp, nocf_split, counts, errors)
        for extra in args.extra:
            import_extra(extra, temp, counts, errors)
        if errors:
            for error in errors:
                print(f'ERROR {error}', file=sys.stderr)
            shutil.rmtree(temp, ignore_errors=True)
            sys.exit(1)
        sources = {
            'hascf': str(args.hascf.resolve()),
            'nocf': str(args.nocf.resolve()),
            'extra': [str(path.resolve()) for path in args.extra],
            'seed': args.seed,
            'val_ratio': args.val_ratio,
            'test_ratio': args.test_ratio,
            'counts': {f'{name}/{split}': count
                       for (name, split), count in sorted(counts.items())},
        }
        (temp / 'sources.json').write_text(
            json.dumps(sources, indent=2, ensure_ascii=False) + '\n')
        (temp / 'run.json').write_text(
            json.dumps({'status': 'complete', 'seed': args.seed}, indent=2) + '\n')
        if args.output.exists():
            shutil.rmtree(args.output)
        temp.replace(args.output)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise

    print('summary:')
    for split in SPLITS:
        total = sum(count for (_, item_split), count in counts.items()
                    if item_split == split)
        print(f'  {split}: {total}')
    print(f'Dataset written to {args.output}')


if __name__ == '__main__':
    main()
