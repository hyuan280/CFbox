"""Run the two-stage detector on screenshots and save JSON/overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cfbox_detector import CFChallengeDetector, ROOT

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def collect_images(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f'Input not found: {path}')
    iterator = path.rglob('*') if recursive else path.glob('*')
    images = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f'No images found in {path}')
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, help='Single screenshot')
    parser.add_argument('--images-dir', type=Path, help='Directory of screenshots')
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--overlay-dir', type=Path)
    parser.add_argument('--json-dir', type=Path)
    parser.add_argument('--json', action='store_true', help='Print one JSON object per line')
    parser.add_argument('--widget-weights', type=Path,
                        default=ROOT / 'runs/widget/weights/best.pt')
    parser.add_argument('--state-weights', type=Path,
                        default=ROOT / 'runs/state/weights/best.pt')
    parser.add_argument('--conf-widget', type=float, default=0.25)
    parser.add_argument('--conf-state', type=float, default=0.25)
    parser.add_argument('--imgsz-widget', type=int, default=1280)
    parser.add_argument('--imgsz-state', type=int, default=640)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if not args.image and not args.images_dir:
        parser.error('Provide --image or --images-dir')
    if args.image and args.images_dir:
        parser.error('Use either --image or --images-dir, not both')
    try:
        images = collect_images(args.image or args.images_dir, args.recursive)
        detector = CFChallengeDetector(
            args.widget_weights, args.state_weights,
            imgsz_widget=args.imgsz_widget, imgsz_state=args.imgsz_state,
            conf_widget=args.conf_widget, conf_state=args.conf_state,
            device=args.device, threads=args.threads)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    if args.overlay_dir:
        args.overlay_dir.mkdir(parents=True, exist_ok=True)
    if args.json_dir:
        args.json_dir.mkdir(parents=True, exist_ok=True)
    for path in images:
        result = detector.predict(path)
        output = {'image': str(path), **result}
        if args.overlay_dir:
            detector.annotate(path, result).save(args.overlay_dir / f'{path.stem}.png')
        if args.json_dir:
            (args.json_dir / f'{path.stem}.json').write_text(
                json.dumps(output, indent=2, ensure_ascii=False))
        if args.json:
            print(json.dumps(output, ensure_ascii=False))
        else:
            center = output['checkbox_center_pct']
            position = '' if not center else f" ({center['x']:.2f}%, {center['y']:.2f}%)"
            print(f"{path}: {output['state']} "
                  f"confidence={output['confidence']}{position}")


if __name__ == '__main__':
    main()
