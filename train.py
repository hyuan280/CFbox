"""Training entry point for either stage of the screenshot detector."""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics'))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['widget', 'state'], default='widget')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'datasets/demo')
    parser.add_argument('--weights', type=Path, default=ROOT / 'models/yolo11n.pt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, help='Default: widget=1280, state=640')
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--device', default='auto',
                        help="auto, cpu, cuda, cuda:0, 0, ... (default: auto)")
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable automatic mixed precision on CUDA')
    parser.add_argument('--name', default=None)
    parser.add_argument('--download-only', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.batch, args.threads, args.imgsz or 640) < 1:
        parser.error('Training sizes, epochs and threads must be positive')
    import torch
    from ultralytics import YOLO, settings
    settings.update({'sync': False, 'weights_dir': str(ROOT / 'models')})
    if hasattr(torch.backends, 'nnpack'):
        torch.backends.nnpack.set_flags(False)
    torch.set_num_threads(args.threads)
    device = args.device
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    elif device != 'cpu' and not torch.cuda.is_available():
        parser.error('CUDA device requested but torch.cuda.is_available() is false')
    print(f'Training device: {device}')
    if not args.download_only:
        run_file = args.dataset / 'run.json'
        if run_file.exists() and json.loads(run_file.read_text())['status'] != 'complete':
            parser.error('Dataset generation is incomplete')
        if not (args.dataset / args.stage / 'data.yaml').is_file():
            parser.error('Dataset data.yaml does not exist')
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    if args.download_only:
        print(f'Weights ready: {args.weights.resolve()}')
        return
    model.train(data=str((args.dataset / args.stage / 'data.yaml').resolve()),
                device=device, epochs=args.epochs,
                imgsz=args.imgsz or (1280 if args.stage == 'widget' else 640),
                batch=args.batch, workers=0, seed=42,
                project=str(ROOT / 'runs'), name=args.name or args.stage,
                exist_ok=False, amp=not args.no_amp and device != 'cpu',
                plots=False,
                fliplr=0, flipud=0, mosaic=0, mixup=0,
                degrees=0, perspective=0)


if __name__ == '__main__':
    main()
