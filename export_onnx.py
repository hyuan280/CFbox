"""Export trained PyTorch weights to ONNX for the deployment image."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics'))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')


def export_model(weights: Path, output: Path, name: str, imgsz: int,
                 opset: int, dynamic: bool) -> dict:
    from ultralytics import YOLO, __version__

    model = YOLO(str(weights))
    names = {int(key): str(value) for key, value in model.names.items()}
    exported = Path(model.export(format='onnx', imgsz=imgsz, opset=opset, batch=1,
                                 dynamic=dynamic, simplify=False, nms=False,
                                 device='cpu'))
    destination = output / f'{name}.onnx'
    if exported.resolve() == destination.resolve():
        pass
    elif destination.exists():
        destination.unlink()
    shutil.move(str(exported), destination)
    return {'file': destination.name, 'imgsz': imgsz, 'dynamic': dynamic,
            'names': names, 'ultralytics_version': __version__}


def verify_model(path: Path, imgsz: int) -> dict:
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1:
        raise RuntimeError(f'{path} must have exactly one input, found {len(inputs)}')
    shape = inputs[0].shape
    if len(shape) != 4 or shape[1] != 3:
        raise RuntimeError(f'{path} input shape is {shape}, expected NCHW')
    if shape[0] not in (1, 'batch', None):
        raise RuntimeError(f'{path} has unsupported batch dimension: {shape[0]}')
    dynamic = any(isinstance(value, str) or value is None for value in shape[2:])
    expected = [1, 3, imgsz, imgsz]
    if not dynamic and list(shape) != expected:
        raise RuntimeError(f'{path} input shape is {shape}, expected {expected}')
    dummy = np.zeros(expected, dtype=np.float32)
    prediction = session.run(None, {inputs[0].name: dummy})[0]
    print(f'{path.name}: input={shape} output={prediction.shape} dynamic={dynamic}')
    return {'input': inputs[0].name, 'output': outputs[0].name,
            'output_shape': list(prediction.shape), 'dynamic': dynamic}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--widget', type=Path,
                        default=ROOT / 'runs/widget/weights/best.pt')
    parser.add_argument('--state', type=Path,
                        default=ROOT / 'runs/state/weights/best.pt')
    parser.add_argument('--output', type=Path, default=ROOT / 'models/onnx')
    parser.add_argument('--imgsz-widget', type=int, default=1280)
    parser.add_argument('--imgsz-state', type=int, default=640)
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--no-dynamic', action='store_true',
                        help='Export fixed square inputs instead of dynamic H/W')
    parser.add_argument('--no-verify', action='store_true')
    args = parser.parse_args()
    if min(args.imgsz_widget, args.imgsz_state, args.opset) < 1:
        parser.error('Image sizes and opset must be positive')
    for path in (args.widget, args.state):
        if not path.is_file():
            sys.exit(f'error: weights not found: {path}')
    args.output.mkdir(parents=True, exist_ok=True)

    dynamic = not args.no_dynamic
    widget = export_model(args.widget, args.output, 'widget',
                          args.imgsz_widget, args.opset, dynamic)
    state = export_model(args.state, args.output, 'state',
                         args.imgsz_state, args.opset, dynamic)
    manifest = {
        'format': 'onnx',
        'opset': args.opset,
        'exported_at': datetime.now(timezone.utc).isoformat(),
        'widget': widget,
        'state': state,
    }
    if not args.no_verify:
        manifest['widget']['runtime'] = verify_model(args.output / widget['file'],
                                                     args.imgsz_widget)
        manifest['state']['runtime'] = verify_model(args.output / state['file'],
                                                    args.imgsz_state)
    (args.output / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(f'ONNX models written to {args.output}')


if __name__ == '__main__':
    main()
