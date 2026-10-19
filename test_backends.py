import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from backends import (LetterboxMeta, decode_output, letterbox, nms,
                      unletterbox, xywh_to_xyxy)
from cfbox_detector import CFChallengeDetector, ROOT

ONNX_WIDGET = ROOT / 'models/onnx/widget.onnx'
ONNX_STATE = ROOT / 'models/onnx/state.onnx'
TORCH_WIDGET = ROOT / 'runs/widget/weights/best.pt'
TORCH_STATE = ROOT / 'runs/state/weights/best.pt'


class BackendUnitTests(unittest.TestCase):
    def test_letterbox_large_image(self):
        tensor, meta = letterbox(Image.new('RGB', (1920, 1080), 'white'), 1280)
        self.assertEqual(tensor.shape, (1, 3, 1280, 1280))
        self.assertAlmostEqual(meta.ratio, 2 / 3, places=5)
        self.assertEqual((meta.resized_width, meta.resized_height), (1280, 720))
        self.assertEqual((meta.left, meta.top), (0, 280))

    def test_letterbox_small_image_upscales(self):
        _, meta = letterbox(Image.new('RGB', (320, 240), 'white'), 1280)
        self.assertEqual(meta.ratio, 4)
        self.assertEqual((meta.resized_width, meta.resized_height), (1280, 960))
        self.assertEqual(meta.top, 160)

    def test_xywh_conversion_does_not_mutate_inputs(self):
        converted = xywh_to_xyxy(np.array([[50, 50, 20, 10]], dtype=np.float32))
        np.testing.assert_allclose(converted, [[40, 45, 60, 55]])

    def test_unletterbox(self):
        meta = LetterboxMeta(2.0, 5, 10, 100, 100, 50, 50)
        mapped = unletterbox(np.array([[25, 30, 45, 50]], dtype=np.float32), meta)
        np.testing.assert_allclose(mapped, [[10, 10, 20, 20]])

    def test_nms_is_class_aware(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]],
                         dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        classes = np.array([0, 0, 1])
        self.assertEqual(nms(boxes, scores, classes, iou_threshold=0.6), [0, 2])

    def test_decode_output(self):
        output = np.zeros((1, 5, 1), dtype=np.float32)
        output[0, :, 0] = [50, 50, 20, 10, 0.9]
        meta = LetterboxMeta(1.0, 0, 0, 100, 100, 100, 100)
        detections = decode_output(output, meta, 0.5, 0.7, 20, [0], 1)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_id, 0)
        self.assertAlmostEqual(detections[0].confidence, 0.9)
        np.testing.assert_allclose(detections[0].box, (40, 45, 60, 55))


@unittest.skipUnless(ONNX_WIDGET.is_file() and ONNX_STATE.is_file()
                     and TORCH_WIDGET.is_file() and TORCH_STATE.is_file(),
                     'torch and ONNX weights are required')
class BackendParityTests(unittest.TestCase):
    def test_unchecked_prediction_matches_torch(self):
        sample = ROOT / 'datasets/demo/widget/images/test/sample_000041.png'
        torch_detector = CFChallengeDetector(TORCH_WIDGET, TORCH_STATE,
                                             backend='torch', threads=2)
        onnx_detector = CFChallengeDetector(ONNX_WIDGET, ONNX_STATE,
                                            backend='onnx',
                                            manifest=ONNX_WIDGET.parent / 'manifest.json',
                                            threads=2)
        torch_result = torch_detector.predict(sample)
        onnx_result = onnx_detector.predict(sample)
        self.assertEqual(torch_result['state'], 'unchecked')
        self.assertEqual(onnx_result['state'], 'unchecked')
        self.assertLess(abs(torch_result['checkbox_center_pct']['x']
                            - onnx_result['checkbox_center_pct']['x']), 1.5)
        self.assertLess(abs(torch_result['checkbox_center_pct']['y']
                            - onnx_result['checkbox_center_pct']['y']), 1.5)


if __name__ == '__main__':
    unittest.main()
