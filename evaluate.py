"""Evaluate the two-stage detector against the generated metadata."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean, median
import sys

from cfbox_detector import (CFChallengeDetector, POSITIVE_STATES, RESULT_STATES, ROOT,
                            box_center, point_in_box)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else None


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def rounded(value, digits=4):
    return None if value is None else round(float(value), digits)


def load_samples(dataset, split):
    samples = []
    for meta_path in sorted((dataset / 'metadata').glob('*.json')):
        meta = json.loads(meta_path.read_text())
        if split != 'all' and meta.get('split') != split:
            continue
        name = meta_path.stem
        image_path = dataset / 'widget/images' / meta['split'] / f'{name}.png'
        crop_path = dataset / 'state/images' / meta['split'] / f'{name}.png'
        if not image_path.is_file():
            raise FileNotFoundError(f'Missing image: {image_path}')
        samples.append({'name': name, 'meta': meta, 'image': image_path,
                        'crop': crop_path})
    return samples


def matrix_json(row_labels, col_labels, matrix):
    return {'rows': list(row_labels), 'cols': list(col_labels),
            'values': [[int(matrix[row][col]) for col in col_labels]
                       for row in row_labels]}


def format_matrix(title, row_labels, col_labels, matrix):
    width = max(12, *(len(label) + 2 for label in col_labels))
    lines = [title, 'gt\\pred'.ljust(12) + ''.join(label.rjust(width) for label in col_labels)]
    for row in row_labels:
        counts = ''.join(str(int(matrix[row][col])).rjust(width) for col in col_labels)
        lines.append(row.ljust(12) + counts)
    return '\n'.join(lines)


def evaluate(args):
    samples = load_samples(args.dataset, args.split)
    if args.limit:
        samples = samples[:args.limit]
    if not samples:
        raise ValueError('No samples matched the selected split')
    detector = CFChallengeDetector(
        args.widget_weights, args.state_weights,
        imgsz_widget=args.imgsz_widget, imgsz_state=args.imgsz_state,
        conf_widget=args.conf_widget, conf_state=args.conf_state,
        threads=args.threads, conflict_margin=args.conflict_margin)
    if args.overlay_dir:
        args.overlay_dir.mkdir(parents=True, exist_ok=True)

    e2e_matrix = defaultdict(Counter)
    oracle_matrix = defaultdict(Counter)
    tp = fp = fn = tn = 0
    widget_ious = []
    e2e_correct = positive_correct = 0
    oracle_correct = positive_total = 0
    exact_correct = 0
    e2e_loc_total = e2e_loc_hits = 0
    oracle_loc_total = oracle_loc_hits = 0
    e2e_errors, e2e_norm_errors = [], []
    oracle_errors, oracle_norm_errors = [], []
    latencies, widget_latencies, state_latencies = [], [], []
    gt_states = Counter()

    for sample in samples:
        meta = sample['meta']
        gt_state = meta['state']
        gt_present = bool(meta['has_cf_challenge'])
        gt_states[gt_state] += 1
        result = detector.predict(sample['image'])
        pred_state = result['state']
        e2e_matrix[gt_state][pred_state] += 1
        latencies.append(result['timing_ms']['total'])
        widget_latencies.append(result['timing_ms']['widget'])
        state_latencies.append(result['timing_ms']['state'])

        if gt_present:
            positive_total += 1
            if result['has_cf_challenge']:
                tp += 1
            else:
                fn += 1
            gt_widget = meta.get('widget_pixels')
            if result['challenges'] and gt_widget:
                widget_ious.append(max(iou(c['widget_box_px'], gt_widget)
                                      for c in result['challenges']))
            if pred_state == gt_state:
                positive_correct += 1
        else:
            if result['has_cf_challenge']:
                fp += 1
            else:
                tn += 1
        if pred_state == gt_state:
            e2e_correct += 1

        gt_checkbox = meta.get('checkbox_pixels')
        if gt_state == 'unchecked' and gt_checkbox:
            e2e_loc_total += 1
            center = result.get('checkbox_center_pct')
            if pred_state == 'unchecked' and center:
                width, height = result['image_size']
                predicted = (center['x'] / 100 * width, center['y'] / 100 * height)
                error = distance(predicted, box_center(gt_checkbox))
                e2e_errors.append(error)
                e2e_norm_errors.append(error / math.hypot(width, height) * 100)
                e2e_loc_hits += int(point_in_box(predicted, gt_checkbox))

        if gt_present:
            if not sample['crop'].is_file():
                raise FileNotFoundError(f"Missing state crop: {sample['crop']}")
            oracle = detector.predict_state(sample['crop'])
            oracle_matrix[gt_state][oracle['state']] += 1
            if oracle['state'] == gt_state:
                oracle_correct += 1
            if gt_state == 'unchecked' and gt_checkbox:
                oracle_loc_total += 1
                local_box = oracle.get('state_box_px')
                if oracle['state'] == 'unchecked' and local_box:
                    offset = meta['crop_pixels'][:2]
                    local_center = box_center(local_box)
                    predicted = (local_center[0] + offset[0],
                                 local_center[1] + offset[1])
                    width, height = result['image_size']
                    error = distance(predicted, box_center(gt_checkbox))
                    oracle_errors.append(error)
                    oracle_norm_errors.append(error / math.hypot(width, height) * 100)
                    oracle_loc_hits += int(point_in_box(predicted, gt_checkbox))

        if gt_state == 'absent':
            exact = pred_state == 'absent'
        elif gt_state == 'unchecked':
            predicted_center = result.get('checkbox_center_pct')
            exact = bool(pred_state == 'unchecked' and predicted_center is not None
                         and gt_checkbox and point_in_box(
                             (predicted_center['x'] / 100 * result['image_size'][0],
                              predicted_center['y'] / 100 * result['image_size'][1]),
                             gt_checkbox))
        else:
            exact = pred_state == gt_state
        exact_correct += int(bool(exact))
        if args.overlay_dir:
            detector.annotate(sample['image'], result).save(
                args.overlay_dir / f"{sample['name']}.png")

    predicted_positive = tp + fp
    metrics = {
        'images': len(samples),
        'positives': positive_total,
        'negatives': len(samples) - positive_total,
        'gt_states': dict(sorted(gt_states.items())),
        'presence': {
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'recall': rounded(safe_div(tp, tp + fn)),
            'false_positive_rate': rounded(safe_div(fp, fp + tn)),
            'precision': rounded(safe_div(tp, predicted_positive)),
        },
        'widget_iou': {
            'count': len(widget_ious),
            'mean': rounded(mean(widget_ious)) if widget_ious else None,
            'median': rounded(median(widget_ious)) if widget_ious else None,
            'at_0_5': sum(value >= 0.5 for value in widget_ious),
            'at_0_5_rate': rounded(safe_div(sum(value >= 0.5 for value in widget_ious),
                                            len(widget_ious))),
        },
        'state_accuracy_end_to_end': {
            'overall': rounded(safe_div(e2e_correct, len(samples))),
            'overall_count': f'{e2e_correct}/{len(samples)}',
            'positives': rounded(safe_div(positive_correct, positive_total)),
            'positives_count': f'{positive_correct}/{positive_total}',
        },
        'state_accuracy_oracle_crop': {
            'accuracy': rounded(safe_div(oracle_correct, positive_total)),
            'count': f'{oracle_correct}/{positive_total}',
        },
        'unchecked_center': {
            'end_to_end': {
                'total': e2e_loc_total,
                'hits': e2e_loc_hits,
                'hit_rate': rounded(safe_div(e2e_loc_hits, e2e_loc_total)),
                'centers_available': len(e2e_errors),
                'mean_error_px': rounded(mean(e2e_errors)) if e2e_errors else None,
                'median_error_px': rounded(median(e2e_errors)) if e2e_errors else None,
                'p95_error_px': rounded(percentile(e2e_errors, 0.95)),
                'mean_error_pct_diagonal': rounded(mean(e2e_norm_errors)) if e2e_norm_errors else None,
            },
            'oracle_crop': {
                'total': oracle_loc_total,
                'hits': oracle_loc_hits,
                'hit_rate': rounded(safe_div(oracle_loc_hits, oracle_loc_total)),
                'centers_available': len(oracle_errors),
                'mean_error_px': rounded(mean(oracle_errors)) if oracle_errors else None,
                'median_error_px': rounded(median(oracle_errors)) if oracle_errors else None,
                'p95_error_px': rounded(percentile(oracle_errors, 0.95)),
                'mean_error_pct_diagonal': rounded(mean(oracle_norm_errors)) if oracle_norm_errors else None,
            },
        },
        'end_to_end_exact': {
            'correct': exact_correct,
            'total': len(samples),
            'rate': rounded(safe_div(exact_correct, len(samples))),
        },
        'latency_ms': {
            'mean': rounded(mean(latencies)) if latencies else None,
            'p50': rounded(percentile(latencies, 0.5)),
            'p95': rounded(percentile(latencies, 0.95)),
            'widget_mean': rounded(mean(widget_latencies)) if widget_latencies else None,
            'state_mean': rounded(mean(state_latencies)) if state_latencies else None,
        },
    }
    report = {
        'config': {
            'dataset': str(args.dataset),
            'split': args.split,
            'widget_weights': str(args.widget_weights),
            'state_weights': str(args.state_weights),
            'conf_widget': args.conf_widget,
            'conf_state': args.conf_state,
            'imgsz_widget': args.imgsz_widget,
            'imgsz_state': args.imgsz_state,
            'conflict_margin': args.conflict_margin,
        },
        'metrics': metrics,
        'confusion_end_to_end': matrix_json(RESULT_STATES, RESULT_STATES, e2e_matrix),
        'confusion_oracle_crop': matrix_json(POSITIVE_STATES,
                                              list(POSITIVE_STATES) + ['unknown'],
                                              oracle_matrix),
    }

    print(f"Dataset: {args.dataset}  split={args.split}  images={len(samples)}")
    if args.split in ('train', 'all'):
        print('WARNING: train/all includes images used for training; not a clean test set.')
    print(f"Presence: TP={tp} FP={fp} FN={fn} TN={tn}  "
          f"recall={metrics['presence']['recall']}  FPR={metrics['presence']['false_positive_rate']}")
    print(f"Widget IoU: mean={metrics['widget_iou']['mean']}  "
          f"median={metrics['widget_iou']['median']}  "
          f"IoU>=0.5={metrics['widget_iou']['at_0_5']}/{metrics['widget_iou']['count']}")
    print(f"State accuracy (end-to-end): "
          f"{metrics['state_accuracy_end_to_end']['overall_count']} overall, "
          f"{metrics['state_accuracy_end_to_end']['positives_count']} positives")
    print(f"State accuracy (oracle crop): "
          f"{metrics['state_accuracy_oracle_crop']['count']}")
    for label in ('end_to_end', 'oracle_crop'):
        loc = metrics['unchecked_center'][label]
        print(f"Unchecked center ({label}): {loc['hits']}/{loc['total']} hits, "
              f"mean error={loc['mean_error_px']} px, "
              f"p95={loc['p95_error_px']} px")
    print(f"End-to-end exact: {exact_correct}/{len(samples)} "
          f"({metrics['end_to_end_exact']['rate']})")
    print(f"Latency: mean={metrics['latency_ms']['mean']} ms  "
          f"p95={metrics['latency_ms']['p95']} ms")
    print()
    print(format_matrix('End-to-end confusion:', RESULT_STATES, RESULT_STATES, e2e_matrix))
    print()
    print(format_matrix('State-only confusion (oracle crop):', POSITIVE_STATES,
                        list(POSITIVE_STATES) + ['unknown'], oracle_matrix))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'datasets/demo')
    parser.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='test')
    parser.add_argument('--widget-weights', type=Path,
                        default=ROOT / 'runs/widget/weights/best.pt')
    parser.add_argument('--state-weights', type=Path,
                        default=ROOT / 'runs/state/weights/best.pt')
    parser.add_argument('--conf-widget', type=float, default=0.25)
    parser.add_argument('--conf-state', type=float, default=0.25)
    parser.add_argument('--imgsz-widget', type=int, default=1280)
    parser.add_argument('--imgsz-state', type=int, default=640)
    parser.add_argument('--conflict-margin', type=float, default=0.15)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--overlay-dir', type=Path)
    parser.add_argument('--json-out', type=Path)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    try:
        evaluate(args)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f'error: {exc}')


if __name__ == '__main__':
    main()
