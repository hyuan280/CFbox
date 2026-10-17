import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from generate import initialize, save_sample, yolo_line


class DatasetTests(unittest.TestCase):
    def test_yolo(self):
        self.assertEqual(yolo_line(0, [20, 10, 60, 30], 100, 100),
                         '0 0.40000000 0.20000000 0.40000000 0.20000000\n')
        for box in [[-1, 0, 1, 1], [0, 0, 0, 1], [0, 0, 101, 1], [0, 0, float('nan'), 1]]:
            with self.assertRaises(ValueError):
                yolo_line(0, box, 100, 100)

    def test_exports_and_pixel_scaling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'dataset'
            initialize(root)
            with self.assertRaises(FileExistsError):
                initialize(root)
            buffer = io.BytesIO()
            Image.new('RGB', (400, 200), 'white').save(buffer, format='PNG')
            annotation = {'state': 'unchecked', 'family': 'portal',
                          'viewport': {'width': 200, 'height': 100},
                          'widget': [20, 10, 180, 80], 'checkbox': [30, 30, 50, 50]}
            save_sample(root, 'positive', 'train', buffer.getvalue(), annotation)
            meta = json.loads((root / 'metadata/positive.json').read_text())
            self.assertEqual(meta['checkbox_center_pct'], {'x': 20, 'y': 40})
            self.assertEqual(meta['widget_pixels'], [40, 20, 360, 160])
            crop = Image.open(root / 'state/images/train/positive.png')
            self.addCleanup(crop.close)
            values = (root / 'state/labels/train/positive.txt').read_text().split()
            cx, cy = float(values[1])*crop.width, float(values[2])*crop.height
            self.assertAlmostEqual(cx + meta['crop_pixels'][0], 80, places=4)
            self.assertAlmostEqual(cy + meta['crop_pixels'][1], 80, places=4)
            for state in ['verified', 'processing', 'error', 'absent']:
                annotation.update(state=state, checkbox=None)
                if state == 'absent':
                    annotation['widget'] = None
                save_sample(root, state, 'val', buffer.getvalue(), annotation)
            self.assertEqual((root / 'widget/labels/val/absent.txt').read_text(), '')
            self.assertFalse((root / 'state/images/val/absent.png').exists())
            self.assertTrue((root / 'state/labels/val/error.txt').read_text().startswith('3 '))


if __name__ == '__main__':
    unittest.main()
