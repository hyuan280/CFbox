"""Compare the torch and ONNX backends on generated screenshots."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from cfbox_detector import CFChallengeDetector, ROOT, point_in_box
from evaluate import iou, load_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'datasets/demo')
    parser.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='all')
    parser.add_argument('--torch-widget', type=Path,
                        default=ROOT / 'runs/widget/weights/best.pt')
    parser.add_argument('--torch-state', type=Path,
                        default=ROOT / 'runs/state/weights/best.pt')
    parser.add_argument('--onnx-widget', type=Path,
                        default=ROOT / 'models/onnx/widget.onnx')
    parser.add_argument('--onnx-state', type=Path,
                        default=ROOT / 'models/onnx/state.onnx')
    parser.add_argument('--manifest', type=Path,
                        default=ROOT / 'models/onnx/manifest.json')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--strict-state', action='store_true',
                        help='Fail if top-level state differs on borderline samples')
    args = parser.parse_args()
    try:
        samples = load_samples(args.dataset, args.split)
        torch_detector = CFChallengeDetector(
            args.torch_widget, args.torch_state, backend='torch',
            threads=args.threads)
        onnx_detector = CFChallengeDetector(
            args.onnx_widget, args.onnx_state, backend='onnx',
            manifest=args.manifest, threads=args.threads)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f'error: {exc}')
    if not samples:
        sys.exit('error: no samples matched the selected split')

    state_mismatches = []
    presence_mismatches = []
    center_errors = []
    center_misses = 0
    iou_values = []
    torch_ms = onnx_ms = 0.0
    for sample in samples:
        torch_result = torch_detector.predict(sample['image'])
        onnx_result = onnx_detector.predict(sample['image'])
        torch_ms += torch_result['timing_ms']['total']
        onnx_ms += onnx_result['timing_ms']['total']
        if torch_result['has_cf_challenge'] != onnx_result['has_cf_challenge']:
            presence_mismatches.append(sample['name'])
        if torch_result['state'] != onnx_result['state']:
            state_mismatches.append((sample['name'], torch_result['state'],
                                     onnx_result['state']))
        if torch_result['challenges'] and onnx_result['challenges']:
            iou_values.append(max(iou(t['widget_box_px'], o['widget_box_px'])
                                  for t in torch_result['challenges']
                                  for o in onnx_result['challenges']))
        gt_checkbox = sample['meta'].get('checkbox_pixels')
        if gt_checkbox:
            centers = [result.get('checkbox_center_pct')
                       for result in (torch_result, onnx_result)]
            if all(centers):
                width, height = onnx_result['image_size']
                points = [(center['x'] / 100 * width, center['y'] / 100 * height)
                          for center in centers]
                center_errors.append(abs(points[0][0] - points[1][0]) +
                                     abs(points[0][1] - points[1][1]))
                center_misses += sum(not point_in_box(point, gt_checkbox)
                                     for point in points)

    print(f'images={len(samples)}  presence mismatches={len(presence_mismatches)}')
    print(f'state mismatches={len(state_mismatches)}')
    print(f'widget IoU mean={sum(iou_values) / len(iou_values):.4f}'
          if iou_values else 'widget IoU mean=n/a')
    print(f'center L1 error mean={sum(center_errors) / len(center_errors):.3f} px'
          if center_errors else 'center error=n/a')
    print(f'center misses={center_misses}')
    print(f'latency torch={torch_ms / len(samples):.1f} ms  '
          f'onnx={onnx_ms / len(samples):.1f} ms')
    for name in presence_mismatches:
        print(f'presence mismatch: {name}')
    for name, torch_state, onnx_state in state_mismatches:
        print(f'state mismatch: {name}: torch={torch_state} onnx={onnx_state}')
    if presence_mismatches or center_misses or (args.strict_state and state_mismatches):
        sys.exit(1)


if __name__ == '__main__':
    main()
