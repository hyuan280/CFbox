"""Execution backends for screenshot detection.

Torch is imported only when a TorchBackend is constructed. The ONNX service
therefore does not need torch, torchvision, or Ultralytics installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from cfbox_detector import Detection


@dataclass(frozen=True)
class LetterboxMeta:
    ratio: float
    left: int
    top: int
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int


def letterbox(image: Image.Image, imgsz: int, padding: int = 114) -> tuple[np.ndarray, LetterboxMeta]:
    """Resize with preserved aspect ratio and pad to a square input."""
    image = image.convert('RGB')
    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError('Image must have positive dimensions')
    ratio = min(imgsz / width, imgsz / height)
    resized_width = max(1, round(width * ratio))
    resized_height = max(1, round(height * ratio))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (imgsz, imgsz), (padding, padding, padding))
    left = round((imgsz - resized_width) / 2 - 0.1)
    top = round((imgsz - resized_height) / 2 - 0.1)
    canvas.paste(resized, (left, top))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    array = np.ascontiguousarray(array.transpose(2, 0, 1)[None])
    return array, LetterboxMeta(ratio, left, top, width, height,
                                resized_width, resized_height)


def unletterbox(boxes: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """Map xyxy boxes from letterboxed model space back to the source image."""
    mapped = boxes.astype(np.float32, copy=True)
    mapped[:, [0, 2]] = (mapped[:, [0, 2]] - meta.left) / meta.ratio
    mapped[:, [1, 3]] = (mapped[:, [1, 3]] - meta.top) / meta.ratio
    mapped[:, [0, 2]] = np.clip(mapped[:, [0, 2]], 0, meta.original_width)
    mapped[:, [1, 3]] = np.clip(mapped[:, [1, 3]], 0, meta.original_height)
    return mapped


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = boxes.astype(np.float32, copy=True)
    x = converted[:, 0].copy()
    y = converted[:, 1].copy()
    width = converted[:, 2].copy()
    height = converted[:, 3].copy()
    converted[:, 0] = x - width / 2
    converted[:, 1] = y - height / 2
    converted[:, 2] = x + width / 2
    converted[:, 3] = y + height / 2
    return converted


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area + areas - intersection, 1e-7)


def nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray,
        iou_threshold: float = 0.7, max_det: int = 20) -> list[int]:
    """Class-aware greedy NMS."""
    keep: list[int] = []
    for class_id in np.unique(classes):
        indices = np.where(classes == class_id)[0]
        order = indices[np.argsort(scores[indices])[::-1]]
        while order.size and len(keep) < max_det:
            current = int(order[0])
            keep.append(current)
            if order.size == 1:
                break
            remaining = order[1:]
            overlap = box_iou(boxes[current], boxes[remaining])
            order = remaining[overlap <= iou_threshold]
    return keep[:max_det]


def decode_output(output: np.ndarray, meta: LetterboxMeta, conf: float,
                  iou: float, max_det: int, classes: Sequence[int],
                  num_classes: int) -> list[Detection]:
    """Decode Ultralytics detection output ``(1, 4+nc, N)``."""
    prediction = np.asarray(output, dtype=np.float32)
    if prediction.ndim == 3:
        prediction = prediction[0]
    if prediction.ndim != 2:
        raise ValueError(f'Unexpected ONNX output shape: {output.shape}')
    if prediction.shape[0] == 4 + num_classes:
        prediction = prediction.T
    elif prediction.shape[1] != 4 + num_classes:
        raise ValueError(f'ONNX output has {prediction.shape[1]} values per box, '
                         f'expected {4 + num_classes}')
    class_scores = prediction[:, 4:4 + num_classes]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_ids)), class_ids]
    mask = scores >= conf
    if classes:
        mask &= np.isin(class_ids, np.asarray(list(classes), dtype=class_ids.dtype))
    if not np.any(mask):
        return []
    boxes = unletterbox(xywh_to_xyxy(prediction[mask, :4]), meta)
    scores = scores[mask]
    class_ids = class_ids[mask]
    selected = nms(boxes, scores, class_ids, iou, max_det)
    return [Detection(int(class_ids[index]), float(scores[index]),
                      tuple(float(value) for value in boxes[index]))
            for index in selected]


class TorchBackend:
    def __init__(self, path: Path, imgsz: int, threads: int, device: str,
                 max_det: int, iou: float):
        import torch
        from ultralytics import YOLO, settings

        settings.update({'sync': False})
        torch.set_num_threads(int(threads))
        self.model = YOLO(str(path))
        self.names = {int(key): str(value) for key, value in self.model.names.items()}
        self.imgsz = imgsz
        self.device = device
        self.max_det = max_det
        self.iou = iou

    def detect(self, image: Image.Image, conf: float, classes: list[int]) -> list[Detection]:
        result = self.model.predict(source=image, imgsz=self.imgsz, conf=conf,
                                    iou=self.iou, device=self.device, verbose=False,
                                    max_det=self.max_det, augment=False,
                                    classes=classes)[0]
        if result.boxes is None:
            return []
        return [Detection(int(box.cls[0].item()), float(box.conf[0].item()),
                          tuple(float(value) for value in box.xyxy[0].tolist()))
                for box in result.boxes]


class OnnxBackend:
    def __init__(self, path: Path, imgsz: int, names: dict[int, str], threads: int,
                 max_det: int, iou: float):
        import onnxruntime as ort

        self.path = path
        self.names = names
        self.imgsz = imgsz
        self.max_det = max_det
        self.iou = iou
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(threads)
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), sess_options=options,
                                            providers=['CPUExecutionProvider'])
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(f'{path} must have exactly one input')
        if list(inputs[0].shape) != [1, 3, imgsz, imgsz]:
            raise ValueError(f'{path} input shape is {inputs[0].shape}, '
                             f'expected [1, 3, {imgsz}, {imgsz}]')
        self.input_name = inputs[0].name
        self.output_name = self.session.get_outputs()[0].name
        warmup = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
        self.session.run([self.output_name], {self.input_name: warmup})

    def detect(self, image: Image.Image, conf: float, classes: list[int]) -> list[Detection]:
        tensor, meta = letterbox(image, self.imgsz)
        output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        return decode_output(output, meta, conf, self.iou, self.max_det,
                             classes, len(self.names))


def create_backend(path: Path, kind: str, imgsz: int, threads: int, device: str,
                   max_det: int, iou: float, names: dict[int, str] | None = None):
    if kind == 'onnx':
        if names is None:
            raise ValueError('ONNX backend requires class names from manifest')
        return OnnxBackend(path, imgsz, names, threads, max_det, iou)
    if kind == 'torch':
        return TorchBackend(path, imgsz, threads, device, max_det, iou)
    raise ValueError(f'Unsupported backend: {kind}')
