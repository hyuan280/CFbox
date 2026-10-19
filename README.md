# CFbox

使用本地 HTML/CSS 模拟验证组件，通过 **Camofox REST API** 访问页面、截取视口并读取 DOM 边界，自动导出两阶段 YOLO 数据集；另含 CPU 训练、ONNX 导出、两阶段推理、HTTP 服务和评估脚本。无需 Playwright 客户端或大模型。

这是合成数据生成器，不是验证破解工具。不访问真实挑战、不提交验证、不自动点击。模拟组件并非官方组件，合成集指标不能代表真实场景准确率。

## 快速开始

需要 Python 3.10+ 和已经运行的 Camofox 服务（默认 `http://localhost:9377`）。

```bash
pip install -r requirements.txt
python3 generate.py --count 50 --output datasets/demo
```

先生成少量数据检查外观和标注，再扩大规模：

```bash
python3 generate.py --count 1000 --seed 42 --output datasets/synthetic-v1
```

输出目录必须尚不存在，避免旧数据残留或覆盖。失败的运行保留已采集文件，`run.json` 中状态为 `incomplete`；重新运行请换输出目录。脚本只关闭自己创建的独立用户会话。

认证通过环境变量 `CAMOFOX_API_KEY` 设置，不会写入数据集。服务地址可使用 `--api` 或 `CAMOFOX_URL`。

## 容器网络

Camofox 浏览器必须能访问生成器的 HTTP 服务。本机回环地址不可达时，绑定可达网卡，并指定浏览器侧可访问的地址：

```bash
python3 generate.py --host 0.0.0.0 --port 8765 \
  --page-base http://HOST_REACHABLE_FROM_CONTAINER:8765 \
  --count 50 --output datasets/container-demo
```

将占位域名替换为容器能访问的宿主机地址。只在可信网络开放该端口，服务仅暴露 `web/` 下的模拟页面，不暴露数据集或项目其他文件。

## 页面预览

```bash
python3 generate.py --serve-only --port 8765
```

浏览器打开 `http://localhost:8765/?seed=42&state=unchecked&family=portal`。

- `state`：`unchecked`、`verified`、`processing`、`error`、`absent`。
- `family`：`portal`、`article`、`dashboard`、`checkout`、`support`、`settings`。
- `seed`：固定随机种子，可复现布局、语言、主题、缩放和位置选择；浏览器版本、字体和运行环境变化仍可能影响像素结果。

## 输出结构

```text
datasets/demo/
  run.json
  metadata/sample_000000.json
  widget/data.yaml
  widget/images/{train,val,test}/
  widget/labels/{train,val,test}/
  state/data.yaml
  state/images/{train,val,test}/
  state/labels/{train,val,test}/
```

第一阶段 `widget`：全图检测 `0: cf_widget`，无组件截图对应空标签文件。

第二阶段 `state`：从原始截图按组件边界增加 8 像素边距裁剪。未勾选框紧贴复选框，其余状态框住组件；无组件样本不生成裁剪图。

| 编号 | 类别 |
| --- | --- |
| 0 | checkbox_unchecked |
| 1 | state_verified |
| 2 | state_processing |
| 3 | state_error |

元数据包含状态、CSS 视口尺寸、原图像素框、裁剪起点和 `checkbox_center_pct`。百分比原点为**页面截图左上角**，位置是复选框中心；没有未勾选复选框时为 `null`。YOLO 标签则使用 0～1 归一化坐标。

截图是当前网页视口，**不包含浏览器工具栏，也不是滚动长图**。程序按实际 PNG 尺寸与 CSS 视口比例换算，支持非 1 的设备像素比例。

## 数据分布

当前包含中英文文案、组件明暗主题、不同缩放与位置、四种视口和普通复选框干扰项。状态按固定计划轮换，约 20% 为无组件页面。布局家族按训练/验证/测试隔离，比例约 80/10/10；小样本集不保证每个集合覆盖全部状态，建议至少生成 50 张。

各家族仍共用部分 CSS 和组件模板，因此这种隔离只减少泄漏，并不证明跨真实网站泛化能力。页面是最小模拟，不是像素级真实复刻。应补充授权采集的真实截图，尤其是独立真实测试集、实际错误状态、多组件、遮挡、整页挑战和浏览器外框。当前生成器只生成完整可见的单组件样本，不覆盖这些情况。

## 测试与后续训练

### 隔离 CPU 训练环境

当前机器已在 `.venv/` 安装 CPU 训练环境，不使用系统 Python 的包。新机器可执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.14.1+cpu torchvision==0.29.1+cpu --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-train.txt
.venv/bin/python train.py --download-only
```

需联网下载 CPU 版 PyTorch（约 196 MB）、torchvision、Ultralytics 及其依赖，和 YOLO11 nano 预训练权重（约 5 MB）。不需要 CUDA、GPU 驱动或本地大模型。环境占用与下载缓存需预留数 GB 空间。Ultralytics 使用 AGPL-3.0/商业许可，产品集成前请评估许可要求。

运行两个阶段：

```bash
.venv/bin/python train.py --stage widget --dataset datasets/demo --epochs 50
.venv/bin/python train.py --stage state --dataset datasets/demo --epochs 50
```

默认 CPU、8 个计算线程、batch=2、数据加载 workers=0。全图默认 1280，局部默认 640；可通过 `--threads`、`--batch`、`--imgsz` 调整。预训练权重下载到 `models/`；训练结果保存在 `runs/<stage>/`，重复运行自动使用新目录。`weights/best.pt` 是验证集最优权重，`weights/last.pt` 是最后一轮权重。首次运行可能额外下载绘图字体。只有权重来自可信来源时才加载 `.pt` 文件。

快速检查环境而非训练可用模型：

```bash
.venv/bin/python train.py --stage widget --epochs 1 --imgsz 320 --name smoke-widget
.venv/bin/python train.py --stage state --epochs 1 --imgsz 320 --name smoke-state
```

50 张合成样本训练出的权重只能检查管线，不能据此评价实际识别效果。

## ONNX 导出

部署镜像只使用 ONNX Runtime。训练完成后，在本地导出两个动态宽高模型（最大边长分别为 1280 和 640）：

```bash
.venv/bin/python -m pip install -r requirements-export.txt
.venv/bin/python export_onnx.py
.venv/bin/python verify_onnx.py --split all
```

输出：

```text
models/onnx/widget.onnx      # 1×3×H×W，最大边长 1280
models/onnx/state.onnx       # 1×3×H×W，最大边长 640
models/onnx/manifest.json
```

`manifest.json` 保存类别名、最大输入尺寸、opset 和导出信息。两个 ONNX 文件约 20 MB，已加入 `.gitignore` 白名单，随代码提交；重新训练后需要重新导出并提交更新。默认导出动态宽高；需要固定正方形输入时可使用 `export_onnx.py --no-dynamic`。

`verify_onnx.py` 会比较 torch 与 ONNX 的组件存在性、状态、组件框 IoU、复选框中心误差和延迟。默认只在组件存在性或中心命中出现差异时失败；`--strict-state` 会把状态差异也视为失败。

## 模型接入

`cfbox_detector.py` 支持 `torch` 和 `onnx` 两种后端。训练和本地评估默认使用 `runs/` 下的 `.pt`，服务默认使用 `models/onnx/`。

本地 torch 单张截图：

```bash
.venv/bin/python predict.py --image datasets/demo/widget/images/test/sample_000041.png --json
```

ONNX 推理：

```bash
.venv/bin/python predict.py \
  --backend onnx \
  --widget-weights models/onnx/widget.onnx \
  --state-weights models/onnx/state.onnx \
  --image datasets/demo/widget/images/test/sample_000041.png \
  --json
```

批量截图并保存 JSON 与可视化结果：

```bash
.venv/bin/python predict.py \
  --images-dir real-screenshots/ \
  --json-dir results/predict \
  --overlay-dir overlays/predict
```

Python 调用：

```python
from cfbox_detector import CFChallengeDetector

detector = CFChallengeDetector(
    "models/onnx/widget.onnx",
    "models/onnx/state.onnx",
    backend="onnx",
)
result = detector.predict("screenshot.png")
print(result["has_cf_challenge"])
print(result["state"])
print(result["checkbox_center_pct"])
```

输出顶层字段：

| 字段 | 含义 |
| --- | --- |
| `has_cf_challenge` | 是否检测到组件 |
| `state` | `absent`、`unchecked`、`verified`、`processing`、`error`、`unknown` |
| `confidence` | 状态置信度；无组件时为 `null` |
| `checkbox_center_pct` | 未勾选复选框中心，相对原图左上角，0～100 |
| `ambiguous` | 是否检测到多个组件 |
| `challenges` | 每个组件框、状态、裁剪范围和中心坐标 |
| `timing_ms` | widget、state 和总推理耗时 |

`unknown` 是保守结果：状态置信度不足、不同状态类别冲突、复选框框过于细长或中心越界时，不会猜测坐标。裁剪时固定外扩 8 像素，与生成训练裁剪的规则一致。

ONNX 使用动态宽高输入，最大边长分别为 widget `1280`、state `640`，并按 stride 32 做矩形 letterbox，以匹配 PyTorch 的推理几何。大图会按比例缩小，小图会按比例放大，不会拉伸或直接裁剪；推理后再把坐标映射回原图。超过 `CFBOX_MAX_PIXELS` 的图片在解码阶段拒绝。

## 评估当前模型

```bash
# ONNX，与部署环境一致
.venv/bin/python evaluate.py \
  --backend onnx \
  --widget-weights models/onnx/widget.onnx \
  --state-weights models/onnx/state.onnx \
  --split test \
  --overlay-dir overlays/test \
  --json-out reports/test-metrics.json

# torch 基线
.venv/bin/python evaluate.py --split test
```

`--split` 可选 `train`、`val`、`test`、`all`。`all` 和 `train` 包含训练图片，结果不能当作独立测试。脚本计算：

- 组件存在性的 TP/FP/FN/TN、召回率和误报率。
- 组件框 IoU。
- 端到端状态准确率和混淆矩阵。
- 仅用标注裁剪图时的状态准确率，用于区分状态模型和 widget 定位误差。
- 未勾选复选框中心命中率、像素误差和相对对角线误差。
- 端到端完全正确率和 CPU 延迟。

当前 demo 的 50 轮训练结果统一放在文末的 [Demo 评估数据](#demo-评估数据50-张训练-50-轮)。

## Docker HTTP 服务

`app.py` 提供 FastAPI 接口，`Dockerfile` 构建 ONNX-only CPU 镜像。镜像默认自带 `models/onnx/`，不安装 torch、torchvision 或 Ultralytics；只使用 OpenCV headless 做与 PyTorch 一致的预处理。实测镜像大小约 **506 MB**。

### 启动

```bash
cp .env.example .env
# 建议在 .env 中设置 CFBOX_API_KEY
docker compose up --build -d
docker compose ps
docker compose logs -f cfbox-api
```

默认监听 `http://localhost:8000`。构建只下载 FastAPI、ONNX Runtime、OpenCV headless、NumPy、Pillow 等依赖，不再下载 196 MB PyTorch。如果 `docker compose up --build` 的构建容器无法解析域名，先手动构建再启动：

```bash
docker build --network=host -t cfbox-api:latest .
docker compose up -d
```

构建不依赖本机 pip 缓存或本地 wheelhouse。

ONNX 模型已随镜像复制到 `/models/onnx`。需要覆盖模型时挂载目录：

```bash
docker run \
  -v /path/to/my-models:/models/onnx:ro \
  -p 8000:8000 \
  cfbox-api:latest
```

也可以在 `docker-compose.yml` 中取消 `volumes` 注释。挂载目录必须包含 `widget.onnx`、`state.onnx` 和 `manifest.json`。启动时会校验文件、类别名和输入尺寸，不匹配会直接失败；`/ready` 在模型加载完成前返回 503。

### 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 存活检查，不需要鉴权，模型卸载后仍返回 200 |
| GET | `/ready` | 模型是否已加载；未加载时返回 503 |
| GET | `/info` | 后端、模型路径、阈值、限制、生命周期和鉴权状态 |
| POST | `/predict` | multipart 上传字段 `image`；模型未加载时自动加载 |
| POST | `/models/load` | 主动加载模型，可重置空闲计时 |
| POST | `/models/unload` | 主动卸载模型并释放内存 |
| GET | `/docs` | OpenAPI 交互文档 |

设置 `CFBOX_API_KEY` 后，`/predict`、`/models/load`、`/models/unload` 支持：

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/ready
curl -s http://localhost:8000/info

curl -s \
  -H "X-API-Key: $CFBOX_API_KEY" \
  -F "image=@screenshot.png" \
  http://localhost:8000/predict | python3 -m json.tool

curl -s -X POST \
  -H "X-API-Key: $CFBOX_API_KEY" \
  http://localhost:8000/models/load

curl -s -X POST \
  -H "X-API-Key: $CFBOX_API_KEY" \
  http://localhost:8000/models/unload
```

也支持 `Authorization: Bearer <key>`。未设置 `CFBOX_API_KEY` 时不鉴权，只适合可信内网。对外部署应放在 TLS 反向代理后，并限制请求频率。

模型生命周期：

- 默认启动时加载模型，空闲 `CFBOX_IDLE_UNLOAD_SECONDS` 秒后自动卸载。
- 设置 `CFBOX_IDLE_UNLOAD_SECONDS=0` 可关闭自动卸载。
- 设置 `CFBOX_LOAD_ON_STARTUP=false` 可让容器启动后保持未加载状态。
- 模型卸载后 `/ready` 返回 503，`/health` 仍返回 200；下一次 `/predict` 或 `POST /models/load` 会重新加载。
- 卸载时会释放模型和 ONNX Runtime 会话，并调用 `gc.collect()` 和 Linux `malloc_trim`。RSS 是否立即下降取决于内存分配器，通常会有一定滞后。

本机容器实测（仅参考）：加载后 RSS 约 260–320 MB，手动卸载后约 110–120 MB，重新加载后约 265 MB。

响应在原有检测结果上增加 `request_id`：

```json
{
  "request_id": "a1b2c3...",
  "has_cf_challenge": true,
  "state": "unchecked",
  "confidence": 0.9957,
  "checkbox_center_pct": {"x": 57.14, "y": 46.62},
  "ambiguous": false,
  "image_size": [800, 800],
  "timing_ms": {"widget": 203.4, "state": 28.4, "total": 231.9}
}
```

常见错误：`401` 鉴权失败，`400 invalid_image` 图片无法解析，`413 image_too_large` 超过上传或像素限制，`503 not_ready` 模型尚未加载，`500 inference_failed` 推理异常。

### 配置和限制

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CFBOX_BACKEND` | `onnx` | 镜像只安装 ONNX 依赖 |
| `CFBOX_WIDGET_WEIGHTS` | `/models/onnx/widget.onnx` | 组件模型 |
| `CFBOX_STATE_WEIGHTS` | `/models/onnx/state.onnx` | 状态模型 |
| `CFBOX_MODEL_MANIFEST` | `/models/onnx/manifest.json` | 类别和输入尺寸 |
| `CFBOX_API_KEY` | 空 | 为空时关闭鉴权 |
| `CFBOX_THREADS` | `8` | ONNX Runtime intra-op 线程数 |
| `CFBOX_CONF_WIDGET` | `0.25` | 组件置信度阈值 |
| `CFBOX_CONF_STATE` | `0.25` | 状态置信度阈值 |
| `CFBOX_CONFLICT_MARGIN` | `0.15` | 状态类别冲突阈值 |
| `CFBOX_IOU` | `0.7` | NMS IoU 阈值 |
| `CFBOX_IDLE_UNLOAD_SECONDS` | `300` | 空闲卸载秒数，0 表示关闭 |
| `CFBOX_LOAD_ON_STARTUP` | `true` | 启动时是否加载模型 |
| `CFBOX_MAX_UPLOAD_MB` | `10` | 单个上传大小上限 |
| `CFBOX_MAX_PIXELS` | `50000000` | 解码后的像素上限 |

最大输入尺寸由 `manifest.json` 中的 `imgsz` 控制，实际推理尺寸按图片比例和 stride 32 计算。需要 1920 最大输入时重新导出 ONNX 并更新 manifest。

服务固定单进程，ONNX Runtime 会话复用，推理请求串行执行，避免 CPU 线程过订阅。需要更高吞吐时应部署多个容器实例，而不是把 `--workers` 调大。

更新自包含模型：

```bash
# 替换 models/onnx/ 下的 widget.onnx、state.onnx、manifest.json
docker compose up --build -d
curl -s http://localhost:8000/ready
```

## 许可证

本项目采用 **GNU Affero General Public License v3.0**，见 [`LICENSE`](LICENSE)。

第三方组件、训练/导出依赖以及由 Ultralytics 训练并导出的 ONNX 模型说明，见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

Ultralytics YOLO 默认使用 AGPL-3.0，其官方说明也把训练模型纳入 AGPL 范围。商业、闭源或 SaaS 部署前，请确认 AGPL 义务或获取 Ultralytics Enterprise License。

## Demo 评估数据（50 张，训练 50 轮）

以下数据只用于验证训练、导出和部署管线，**不是真实场景 benchmark**。

### 数据与训练配置

| 项目 | 配置 |
| --- | --- |
| 数据集 | `datasets/demo`，50 张合成截图 |
| 划分 | train 40 / val 5 / test 5 |
| 状态分布 | absent、unchecked、verified、processing、error 各 10 张 |
| widget 模型 | YOLO11n，50 epochs，`imgsz=1280`，CPU |
| state 模型 | YOLO11n，50 epochs，`imgsz=640`，CPU |
| 部署模型 | ONNX，opset 17，FP32，动态矩形输入 |

### ONNX 检测结果

| 数据 | 组件召回 | 无组件误报 | 组件 IoU | 状态 top-1 | 中心命中 | 中心误差 | 端到端完全正确 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test，5 张 | 4/4 | 0/1 | 0.932 | 2/4 | 1/1 | 0.31 px | 3/5 |
| all，50 张 | 40/40 | 0/10 | 0.948 | 18/40 | 10/10 | 0.49 px | 28/50 |

test 只有 5 张，比例没有统计意义；`all` 包含训练图片，也不是独立测试集。

### 状态混淆矩阵（all，50 张）

| GT \ Pred | unchecked | verified | processing | error | unknown |
| --- | ---: | ---: | ---: | ---: | ---: |
| verified | 0 | 0 | 0 | 0 | 10 |
| processing | 0 | 1 | 0 | 0 | 9 |
| error | 0 | 0 | 0 | 8 | 2 |

`unchecked` 10/10 正确，`absent` 10/10 正确；`verified` 全部无法确定，`processing` 多数无法确定。

### torch / ONNX 一致性（all，50 张）

| 指标 | 结果 |
| --- | --- |
| 组件存在性差异 | 0/50 |
| 状态差异 | 0/50 |
| 组件框 IoU | 1.000 |
| 复选框中心 L1 误差 | 0.086 px |
| 中心命中失败 | 0 |

### 解读

- 组件检测和未勾选复选框定位已经可用。
- **状态识别目前不可用**：`verified`、`processing` 经常返回 `unknown`，原因是 50 张数据太少，且不同状态的检测框高度重叠。
- 状态模型的 YOLO mAP 可能不低，但 mAP 不能反映互斥状态的 top-1 分类能力，应以状态准确率和混淆矩阵为准。
- 下一步应扩充到数百张并加入真实截图；若状态仍然混淆，建议改为“裁剪图单标签分类 + 复选框检测”的双头方案。

### 复现

```bash
.venv/bin/python -m unittest -v
.venv/bin/python evaluate.py --backend onnx \
  --widget-weights models/onnx/widget.onnx \
  --state-weights models/onnx/state.onnx \
  --split all
.venv/bin/python verify_onnx.py --split all
```

CPU 全图训练可能耗时较长。检查真实截图上的组件召回率、状态混淆和中心点命中率，再决定是否扩大训练。
