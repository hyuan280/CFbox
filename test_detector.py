import unittest

from PIL import Image

from cfbox_detector import (Detection, annotate_result, box_center, box_to_pct,
                            expand_box, point_in_box, resolve_state,
                            valid_checkbox_box)

STATE_BY_CLASS = {0: 'unchecked', 1: 'verified', 2: 'processing', 3: 'error'}


class DetectorUnitTests(unittest.TestCase):
    def test_resolve_single_state(self):
        top = Detection(1, 0.8, (1, 2, 3, 4))
        state, chosen, reason = resolve_state([top], STATE_BY_CLASS, 0.15)
        self.assertEqual((state, chosen, reason), ('verified', top, None))

    def test_resolve_same_class_keeps_best(self):
        low = Detection(0, 0.4, (0, 0, 1, 1))
        high = Detection(0, 0.9, (0, 0, 2, 2))
        state, chosen, reason = resolve_state([low, high], STATE_BY_CLASS, 0.15)
        self.assertEqual(state, 'unchecked')
        self.assertEqual(chosen, high)
        self.assertIsNone(reason)

    def test_resolve_conflict_is_unknown(self):
        state, chosen, reason = resolve_state(
            [Detection(0, 0.50, (0, 0, 1, 1)), Detection(1, 0.40, (0, 0, 2, 2))],
            STATE_BY_CLASS, 0.15)
        self.assertEqual((state, chosen, reason), ('unknown', None,
                                                   'conflicting_state_detections'))

    def test_resolve_without_detection(self):
        self.assertEqual(resolve_state([], STATE_BY_CLASS, 0.15),
                         ('unknown', None, 'no_state_detection'))

    def test_box_helpers(self):
        self.assertEqual(box_center((10, 20, 30, 40)), (20, 30))
        self.assertEqual(expand_box((2, 3, 98, 97), 8, 100, 100), (0, 0, 100, 100))
        self.assertEqual(box_to_pct((20, 10, 60, 30), 100, 100),
                         {'x1': 20.0, 'y1': 10.0, 'x2': 60.0, 'y2': 30.0})
        self.assertTrue(point_in_box((10, 10), (10, 10, 20, 20)))
        self.assertFalse(point_in_box((9.9, 10), (10, 10, 20, 20)))

    def test_checkbox_geometry(self):
        self.assertTrue(valid_checkbox_box((22, 31.6, 52, 60.6)))
        self.assertFalse(valid_checkbox_box((0, 50.8, 9.2, 87.9)))
        self.assertFalse(valid_checkbox_box((0, 0, 3, 10)))
        self.assertTrue(valid_checkbox_box((0, 0, 4, 8), aspect_range=(0.4, 2.5)))

    def test_annotate_result(self):
        image = Image.new('RGB', (200, 100), 'white')
        result = {'challenges': [{'state': 'unchecked', 'confidence': 0.9,
                                  'widget_box_px': [20, 20, 120, 70],
                                  'checkbox_center_pct': {'x': 30.0, 'y': 45.0}}]}
        output = annotate_result(image, result)
        self.assertEqual(output.size, (200, 100))
        self.assertEqual(output.mode, 'RGB')
        self.assertNotEqual(output.getpixel((20, 20)), (255, 255, 255))
        absent = annotate_result(image, {'challenges': []})
        self.assertEqual(absent.size, (200, 100))


if __name__ == '__main__':
    unittest.main()
