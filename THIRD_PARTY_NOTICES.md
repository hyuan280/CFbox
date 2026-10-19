# Third-Party Notices

CFbox is licensed under the GNU Affero General Public License v3.0. See
[`LICENSE`](LICENSE). This file summarizes third-party components and model
artifacts used by the project. It is informational and is not legal advice.

## Runtime dependencies (Docker image)

| Component | Version | License |
| --- | --- | --- |
| FastAPI | 0.136.3 | MIT |
| Uvicorn | 0.49.0 | BSD-3-Clause |
| Starlette | 1.9.0 | BSD-3-Clause |
| Pydantic | 2.14.1 | MIT |
| python-multipart | 0.0.32 | Apache-2.0 |
| Pillow | 12.3.0 | MIT-CMU |
| NumPy | 2.4.6 | BSD-3-Clause and bundled licenses |
| ONNX Runtime | 1.26.0 | MIT |

The Docker image does not include PyTorch, torchvision, or Ultralytics.

## Training and ONNX export dependencies

These packages are used by the local training/export workflow and are not
installed in the runtime image.

| Component | Version | License |
| --- | --- | --- |
| Ultralytics YOLO | 8.4.178 | AGPL-3.0 |
| PyTorch | 2.14.1+cpu | BSD-3-Clause and bundled licenses |
| torchvision | 0.29.1+cpu | BSD-3-Clause |
| ONNX | 1.23.2 | Apache-2.0 |
| ONNX Runtime | 1.26.0 | MIT |

Ultralytics licenses its code and trained models under AGPL-3.0 by default, or
under a separate Enterprise License. See:

- <https://github.com/ultralytics/ultralytics/blob/main/LICENSE>
- <https://www.ultralytics.com/license>

## Model artifacts

`models/onnx/widget.onnx` and `models/onnx/state.onnx` were exported from
YOLO11n detection weights trained with Ultralytics YOLO. Ultralytics' public
licensing FAQ states that trained models are covered by AGPL-3.0. If you
distribute these model artifacts or a container containing them, comply with
AGPL-3.0 or obtain an Ultralytics Enterprise License.

`models/onnx/manifest.json` records the Ultralytics version, opset, input
sizes, class names, and export time for traceability.

## Data generation

The synthetic screenshot generator uses the Camofox REST API to control a local
browser. Camofox and its browser engine are separate third-party components and
are not bundled in the runtime Docker image.
