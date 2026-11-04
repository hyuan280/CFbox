"""Add hard negatives and targeted generated train samples to a dataset copy."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')


def safe_prefix(name: str) -> str:
    return re.sub(r'[^0-9A-Za-z]+', '_', name).strip('_') + '_'


def find_image(directory: Path, name: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        path = directory / f'{name}{suffix}'
        if path.is_file():
            return path
    return None


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f'directory not found: {directory}')
    return sorted(path for path in directory.iterdir()
                  if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def save_rgb(source: Path, destination: Path):
    with Image.open(source) as image:
        image.convert('RGB').save(destination)


def add_negatives(directory: Path, root: Path, prefix: str, dry_run: bool,
                  counts: dict[str, int], errors: list[str]):
    image_dir = root / 'widget/images/train'
    label_dir = root / 'widget/labels/train'
    for source in image_files(directory):
        name = prefix + source.stem
        meta_path = root / 'metadata' / f'{name}.json'
        if meta_path.exists():
            counts['negative_skipped'] += 1
            continue
        if dry_run:
            counts['negative_added'] += 1
            continue
        try:
            save_rgb(source, image_dir / f'{name}.png')
            (label_dir / f'{name}.txt').write_text('')
            with Image.open(source) as image:
                size = list(image.size)
            metadata = {
                'split': 'train',
                'state': 'absent',
                'has_cf_challenge': False,
                'widget_pixels': None,
                'checkbox_pixels': None,
                'image_size': size,
                'source': 'hard-negative',
                'augmentation_source': str(directory),
            }
            meta_path.write_text(json.dumps(metadata, indent=2) + '\n')
            counts['negative_added'] += 1
        except Exception as exc:
            errors.append(f'{source}: {exc}')


def add_extra_train(extra: Path, root: Path, prefix: str, dry_run: bool,
                    counts: dict[str, int], errors: list[str]):
    run_path = extra / 'run.json'
    if run_path.is_file():
        try:
            run = json.loads(run_path.read_text())
            if run.get('status') != 'complete':
                errors.append(f'{extra}: generation is not complete')
                return
        except json.JSONDecodeError as exc:
            errors.append(f'{extra}: invalid run.json: {exc}')
            return
    metadata_dir = extra / 'metadata'
    if not metadata_dir.is_dir():
        errors.append(f'{extra}: metadata directory missing')
        return
    for meta_path in sorted(metadata_dir.glob('*.json')):
        old_name = meta_path.stem
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f'{extra.name}/{old_name}: invalid JSON: {exc}')
            continue
        if meta.get('split') != 'train':
            continue
        name = prefix + old_name
        target_meta = root / 'metadata' / f'{name}.json'
        if target_meta.exists():
            counts['extra_skipped'] += 1
            continue
        image_path = find_image(extra / 'widget/images/train', old_name)
        if image_path is None:
            errors.append(f'{extra.name}/{old_name}: widget image missing')
            continue
        state_image = extra / 'state/images/train' / f'{old_name}.png'
        state_label = extra / 'state/labels/train' / f'{old_name}.txt'
        if meta.get('has_cf_challenge') and (not state_image.is_file() or not state_label.is_file()):
            errors.append(f'{extra.name}/{old_name}: state crop or label missing')
            continue
        if dry_run:
            counts['extra_added'] += 1
            continue
        try:
            save_rgb(image_path, root / 'widget/images/train' / f'{name}.png')
            widget_label = extra / 'widget/labels/train' / f'{old_name}.txt'
            (root / 'widget/labels/train' / f'{name}.txt').write_text(
                widget_label.read_text() if widget_label.is_file() else '')
            if meta.get('has_cf_challenge'):
                save_rgb(state_image, root / 'state/images/train' / f'{name}.png')
                (root / 'state/labels/train' / f'{name}.txt').write_text(
                    state_label.read_text())
            meta = {**meta, 'split': 'train',
                    'augmentation_source': str(extra)}
            target_meta.write_text(json.dumps(meta, indent=2) + '\n')
            counts['extra_added'] += 1
        except Exception as exc:
            errors.append(f'{extra.name}/{old_name}: {exc}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True,
                        help='Target dataset copy, e.g. datasets/real-v1-ft')
    parser.add_argument('--negatives', type=Path, nargs='*', default=[],
                        help='No-CF image directories to add as absent train samples')
    parser.add_argument('--extra-train', type=Path, nargs='*', default=[],
                        help='Generated datasets whose train split should be added')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    root = args.dataset
    if not (root / 'metadata').is_dir():
        sys.exit(f'error: not a dataset: {root}')
    for directory in (root / 'widget/images/train', root / 'widget/labels/train',
                      root / 'state/images/train', root / 'state/labels/train'):
        directory.mkdir(parents=True, exist_ok=True)

    counts = {'negative_added': 0, 'negative_skipped': 0,
              'extra_added': 0, 'extra_skipped': 0}
    errors: list[str] = []
    prefixes: dict[str, Path] = {}
    for directory in args.negatives:
        prefix = safe_prefix(directory.name)
        if prefix in prefixes:
            errors.append(f'{directory}: prefix collides with {prefixes[prefix]}')
            continue
        prefixes[prefix] = directory
        add_negatives(directory, root, prefix, args.dry_run, counts, errors)
    for extra in args.extra_train:
        prefix = safe_prefix(extra.name)
        if prefix in prefixes:
            errors.append(f'{extra}: prefix collides with {prefixes[prefix]}')
            continue
        prefixes[prefix] = extra
        add_extra_train(extra, root, prefix, args.dry_run, counts, errors)

    for error in errors:
        print(f'ERROR {error}', file=sys.stderr)
    if errors:
        sys.exit(1)
    if not args.dry_run:
        report = {
            'dataset': str(root),
            'negatives': [str(path) for path in args.negatives],
            'extra_train': [str(path) for path in args.extra_train],
            'counts': counts,
        }
        (root / 'augmentations.json').write_text(
            json.dumps(report, indent=2) + '\n')
    print(' '.join(f'{key}={value}' for key, value in counts.items()))
    if args.dry_run:
        print('dry run only; no files written')


if __name__ == '__main__':
    main()
