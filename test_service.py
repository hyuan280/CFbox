import os
from pathlib import Path
import unittest

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent
WIDGET = ROOT / 'models/onnx/widget.onnx'
STATE = ROOT / 'models/onnx/state.onnx'
MANIFEST = ROOT / 'models/onnx/manifest.json'
API_KEY = 'test-secret'

os.environ['CFBOX_WIDGET_WEIGHTS'] = str(WIDGET)
os.environ['CFBOX_STATE_WEIGHTS'] = str(STATE)
os.environ['CFBOX_MODEL_MANIFEST'] = str(MANIFEST)
os.environ['CFBOX_BACKEND'] = 'onnx'
os.environ['CFBOX_API_KEY'] = API_KEY
os.environ['CFBOX_THREADS'] = '2'
os.environ['CFBOX_MAX_UPLOAD_MB'] = '2'
os.environ['CFBOX_LOG_LEVEL'] = 'WARNING'

from app import app  # noqa: E402


@unittest.skipUnless(WIDGET.is_file() and STATE.is_file() and MANIFEST.is_file(),
                     'ONNX models are not available')
class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = TestClient(app)
        cls.client = cls.context.__enter__()
        cls.headers = {'X-API-Key': API_KEY}
        cls.unchecked = (ROOT / 'datasets/demo/widget/images/test/sample_000041.png').read_bytes()
        cls.absent = (ROOT / 'datasets/demo/widget/images/test/sample_000031.png').read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def test_health_and_ready_do_not_require_auth(self):
        health = self.client.get('/health')
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()['ready'])
        self.assertEqual(self.client.get('/ready').status_code, 200)

    def test_info_reports_auth(self):
        response = self.client.get('/info')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['backend'], 'onnx')
        self.assertTrue(response.json()['auth_required'])

    def test_predict_requires_key(self):
        response = self.client.post('/predict', files={
            'image': ('sample.png', self.unchecked, 'image/png')})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()['detail']['code'], 'unauthorized')

    def test_predict_accepts_bearer_key(self):
        response = self.client.post(
            '/predict', files={'image': ('sample.png', self.unchecked, 'image/png')},
            headers={'Authorization': f'Bearer {API_KEY}'})
        self.assertEqual(response.status_code, 200)

    def test_predict_unchecked(self):
        response = self.client.post(
            '/predict', files={'image': ('sample.png', self.unchecked, 'image/png')},
            headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers['X-Request-ID'])
        payload = response.json()
        self.assertTrue(payload['has_cf_challenge'])
        self.assertEqual(payload['state'], 'unchecked')
        self.assertIsNotNone(payload['checkbox_center_pct'])
        self.assertGreater(payload['checkbox_center_pct']['x'], 0)
        self.assertLess(payload['checkbox_center_pct']['x'], 100)

    def test_predict_absent(self):
        response = self.client.post(
            '/predict', files={'image': ('absent.png', self.absent, 'image/png')},
            headers=self.headers)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload['has_cf_challenge'])
        self.assertEqual(payload['state'], 'absent')
        self.assertIsNone(payload['checkbox_center_pct'])

    def test_invalid_image(self):
        response = self.client.post(
            '/predict', files={'image': ('bad.png', b'not an image', 'image/png')},
            headers=self.headers)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()['detail']['code'], 'invalid_image')

    def test_upload_limit(self):
        response = self.client.post(
            '/predict', files={'image': ('big.png', b'x' * (2 * 1024 * 1024 + 1),
                                        'image/png')},
            headers=self.headers)
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()['detail']['code'], 'image_too_large')


if __name__ == '__main__':
    unittest.main()
