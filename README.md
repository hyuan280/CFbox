# CFbox

使用本地 HTML/CSS 模拟验证组件，通过 **Camofox REST API** 访问页面、截取视口并读取 DOM 边界，自动导出两阶段 YOLO 数据集；另含 CPU/GPU 训练、ONNX 导出、两阶段推理、HTTP 服务和评估脚本。无需 Playwright 客户端或大模型。

这是合成数据生成器，不是验证破解工具。不访问真实挑战、不提交验证、不自动点击。模拟组件并非官方组件，合成集指标不能代表真实场景准确率。

## 快速使用

### 使用 GHCR 镜像

```bash
docker pull ghcr.io/hyuan280/cfbox:latest

docker run -d --name cfbox-api \
  --restart unless-stopped \
  -p 8000:8000 \
  -e CFBOX_API_KEY=change-me \
  ghcr.io/hyuan280/cfbox:latest
```

也可以使用 Compose：

```bash
cp .env.example .env
# 在 .env 中设置 CFBOX_API_KEY
docker compose pull
docker compose up -d
```

镜像包含 ONNX 模型，不依赖本地权重。端口冲突时修改 `-p` 左侧；对外部署建议使用 TLS 反向代理并设置 `CFBOX_API_KEY`。

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/ready
```

### 从源码构建

```bash
git clone https://github.com/hyuan280/CFbox.git
cd CFbox
cp .env.example .env
# 在 .env 中设置 CFBOX_API_KEY
docker build --network=host -t ghcr.io/hyuan280/cfbox:latest .
docker compose up -d
```

`--network=host` 仅用于构建容器无法解析域名的情况。Compose 使用 `ghcr.io/hyuan280/cfbox:latest`；本地构建时会用同名镜像覆盖本地缓存。

### 覆盖模型

默认模型位于镜像内 `/models/onnx`。需要替换时挂载目录：

```bash
docker run -d --name cfbox-api \
  -p 8000:8000 \
  -e CFBOX_API_KEY=change-me \
  -v /path/to/my-models:/models/onnx:ro \
  ghcr.io/hyuan280/cfbox:latest
```

挂载目录必须包含 `widget.onnx`、`state.onnx` 和 `manifest.json`。

### 镜像发布

推送 `v*` 标签后，`.github/workflows/docker-publish.yml` 会自动构建并发布 GHCR 镜像。

## API 使用

服务默认监听 `http://localhost:8000`。设置 `CFBOX_API_KEY` 后，`/predict`、`/models/load`、`/models/unload` 需要鉴权。

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
docker build --network=host -t ghcr.io/hyuan280/cfbox:latest .
docker compose up -d
curl -s http://localhost:8000/ready
```

## 开发

### 数据生成

#### 快速开始

需要 Python 3.10+ 和已经运行的 Camofox 服务（默认 `http://localhost:9377`）。

```bash
pip install -r requirements.txt
python3 generate.py --count 50 --output datasets/demo
```

先生成少量数据检查外观和标注，再扩大规模：

```bash
python3 generate.py --count 1000 --seed 42 --output datasets/synthetic-v1
```

输出目录必须尚不存在，避免旧数据残留或覆盖。失败的运行保留已采集文件，`run.json` 中状态为 `incomplete`。用同一个 `--output` 加 `--resume` 可以继续，已完成的样本会跳过；Camofox 返回 500 或页面超时的样本会自动重试并重建标签页。`--retries` 和 `--retry-delay` 可调整重试次数和初始间隔。脚本只关闭自己创建的独立用户会话。

认证通过环境变量 `CAMOFOX_API_KEY` 设置，不会写入数据集。服务地址可使用 `--api` 或 `CAMOFOX_URL`。

#### 容器网络

Camofox 浏览器必须能访问生成器的 HTTP 服务。本机回环地址不可达时，绑定可达网卡，并指定浏览器侧可访问的地址：

```bash
python3 generate.py --host 0.0.0.0 --port 8765 \
  --page-base http://HOST_REACHABLE_FROM_CONTAINER:8765 \
  --count 50 --output datasets/container-demo
```

将占位域名替换为容器能访问的宿主机地址。只在可信网络开放该端口，服务仅暴露 `web/` 下的模拟页面，不暴露数据集或项目其他文件。

#### 页面预览

```bash
python3 generate.py --serve-only --port 8765
```

浏览器打开 `http://localhost:8765/?seed=42&state=unchecked&family=portal`。

- `state`：`unchecked`、`verified`、`processing`、`error`、`absent`。
- `family`：`portal`、`article`、`dashboard`、`checkout`、`support`、`settings`、`table`、`list`、`form`、`dialog`、`sidebar`、`grid`。其中 `table`、`list`、`form`、`dialog`、`sidebar`、`grid` 会把组件嵌入表格、列表、表单、对话框、侧栏或卡片网格中。
- `seed`：固定随机种子，可复现布局、语言、主题、缩放和位置选择；浏览器版本、字体和运行环境变化仍可能影响像素结果。

#### 真实背景合成

可以把**不含 CF 组件**的真实截图作为背景，生成时随机叠加不同状态、尺寸和位置的挑战组件：

```bash
python3 generate.py \
  --real-images real-backgrounds \
  --count 2000 --seed 42 \
  --output datasets/mixed-v1
```

`--real-images` 目录支持 PNG/JPG/JPEG/WebP/BMP，默认所有样本都使用真实背景。可用 `--real-ratio 0.5` 混合真实背景和纯合成页面。

行为说明：

- 真实截图会按 split 分区，避免同一背景同时出现在 train 和 test；图片少于 3 张时全部进入 train，val/test 回退到合成页面。
- 视口尺寸按截图宽高比设置，默认最长边缩放到 1920，可用 `--real-max-size 0` 关闭。
- 使用真实背景时会隐藏内置页面、干扰按钮和侧栏，只保留叠加的 CF 组件。
- 元数据会记录 `source: real-background` 和 `background_image`，方便追踪来源。
- 背景图本身不能包含 CF 挑战框，否则会产生重复组件和错误标注。
- `--host 0.0.0.0` 会把该目录中的背景图暴露给网络，只应在可信网络使用。

#### 合并真实数据

`import_real_dataset.py` 把 `hascf`、`nocf` 和已完成的生成集合并成 train/val/test 数据集：

```bash
python3 import_real_dataset.py \
  --output datasets/real-v1 \
  --hascf images/hascf \
  --nocf images/nocf \
  --extra datasets/gen-altcha datasets/gen-backgrounds datasets/gen-synthetic \
  --seed 42
```

脚本会先完整校验所有标注、图片、state 裁剪和生成集的 `run.json`，全部通过后才在临时目录构建并替换目标目录；校验失败不会产生半成品。目标目录已存在时需要加 `--force`。生成集的 `run.json` 状态不是 `complete` 时不会合并。

#### 输出结构

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

#### 数据分布

当前包含中英文文案、组件明暗主题、不同宽度/高度/宽高比、不同缩放与位置、四种视口和大量干扰项：普通复选框、单选按钮、开关、方框图标按钮、圆形按钮、状态点和头像块。组件既可能独立浮动，也会嵌入表格、列表、表单、对话框、侧栏和卡片网格中。状态按固定计划轮换，约 20% 为无组件页面，这些页面同样包含干扰按钮，用于训练 widget 模型抑制误报。布局家族按训练/验证/测试隔离，比例约 80/10/10；小样本集不保证每个集合覆盖全部状态，建议至少生成 50 张。

各家族仍共用部分 CSS 和组件模板，因此这种隔离只减少泄漏，并不证明跨真实网站泛化能力。页面是最小模拟，不是像素级真实复刻。应补充授权采集的真实截图，尤其是独立真实测试集、实际错误状态、多组件、遮挡、整页挑战和浏览器外框。当前生成器只生成完整可见的单组件样本，不覆盖这些情况。

### 训练环境

使用单独的虚拟环境，不污染系统 Python。CPU 环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.14.1+cpu torchvision==0.29.1+cpu --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-train-cpu.txt
.venv/bin/python train.py --download-only
```

GPU 环境先安装与显卡匹配的 PyTorch，再安装 `requirements-train-gpu.txt`（该文件不含 torch，避免覆盖厂商专用版本）。NVIDIA CUDA 示例：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.14.1+cu130 torchvision==0.29.1+cu130 --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -m pip install -r requirements-train-gpu.txt
.venv/bin/python train.py --download-only
```

AMD ROCm 使用对应的 ROCm PyTorch 索引，其余步骤相同：

```bash
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/rocmX.Y
.venv/bin/python -m pip install -r requirements-train-gpu.txt
```

CUDA/ROCm 版本按本机驱动和 PyTorch 支持矩阵选择；驱动过旧时改用对应的 cu126/cu128 或 ROCm 版本索引。GPU 版 requirements 不能跨厂商通用，因为 torch/torchvision 必须来自 NVIDIA CUDA 或 AMD ROCm 各自的索引。需联网下载 PyTorch、torchvision、Ultralytics 及其依赖，和 YOLO11 nano 预训练权重（约 5 MB）。环境占用与下载缓存需预留数 GB 空间。Ultralytics 使用 AGPL-3.0/商业许可，产品集成前请评估许可要求。

### 模型训练

运行两个阶段：

```bash
.venv/bin/python train.py --stage widget --dataset datasets/demo --epochs 50
.venv/bin/python train.py --stage state --dataset datasets/demo --epochs 50
```

默认 `--device auto`：有可用 CUDA 时使用 `cuda:0`，否则使用 CPU。可用 `--device cpu`、`--device 0` 等显式指定；CUDA 上默认启用 AMP，可用 `--no-amp` 关闭。CPU 默认 8 个计算线程、batch=2、数据加载 workers=0；GPU 训练可提高 `--batch`。全图默认 1280，局部默认 640；可通过 `--threads`、`--batch`、`--imgsz` 调整。预训练权重下载到 `models/`；训练结果保存在 `runs/<stage>/`，重复运行自动使用新目录。`weights/best.pt` 是验证集最优权重，`weights/last.pt` 是最后一轮权重。首次运行可能额外下载绘图字体。只有权重来自可信来源时才加载 `.pt` 文件。

快速检查环境而非训练可用模型：

```bash
.venv/bin/python train.py --stage widget --epochs 1 --imgsz 320 --name smoke-widget
.venv/bin/python train.py --stage state --epochs 1 --imgsz 320 --name smoke-state
```

50 张合成样本训练出的权重只能检查管线，不能据此评价实际识别效果。

### ONNX 导出

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

用于导出的最终训练权重也已纳入 Git：

```text
runs/widget/weights/best.pt
runs/state/weights/best.pt
```

这两份权重来自 `widget-v2`、`state-v2` 训练，并已同步到默认路径；对应的 ONNX 模型在 `models/onnx/`。`last.pt`、smoke 训练权重和其他运行产物仍然不提交。

`verify_onnx.py` 会比较 torch 与 ONNX 的组件存在性、状态、组件框 IoU、复选框中心误差和延迟。默认只在组件存在性或中心命中出现差异时失败；`--strict-state` 会把状态差异也视为失败。

### 本地推理

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

### 模型评估

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

v2 模型在 `datasets/real-v1` 上的评估结果统一放在文末的 [v2 模型与真实混合数据评估](#v2-模型与真实混合数据评估)。

### v2 模型与真实混合数据评估

最终模型为 `widget-v2` 和 `state-v2`，已同步到默认权重路径并导出 ONNX：

```text
runs/widget-v2/weights/best.pt     # 训练产物
runs/state-v2/weights/best.pt
runs/widget/weights/best.pt        # 默认路径，内容与 widget-v2 一致
runs/state/weights/best.pt         # 默认路径，内容与 state-v2 一致
models/onnx/widget.onnx
models/onnx/state.onnx
models/onnx/manifest.json
```

评估使用本地 `datasets/real-v1`，该数据集**不随仓库提交**，共 5000 张：

- 划分：train 3973 / val 501 / test 526
- 真实背景 3100 张：普通背景 3000 张 + ALTCHA 背景 100 张
- 纯合成页面 1649 张
- 真实有 CF 页面 37 张
- 真实无 CF 页面 214 张
- 每个生成集约 20% 为 `absent`，包含按钮、复选框、ALTCHA 等难负样本

#### 训练与导出配置

| 项目 | 配置 |
| --- | --- |
| widget 模型 | YOLO11n，50 epochs，`imgsz=1280`，batch 64，GPU AMP |
| state 模型 | YOLO11n，50 epochs，`imgsz=640`，batch 64，GPU AMP |
| 部署模型 | ONNX，opset 17，FP32，动态宽高输入 |
| 默认阈值 | `conf_widget=0.25`，`conf_state=0.25`，`conflict_margin=0.15`，`iou=0.7` |

#### test 结果（526 张）

| 后端 | Presence | Widget IoU | 状态端到端 | 状态 oracle crop | 未勾选中心 | 完全正确 | 延迟 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| torch | TP 388 / FP 0 / FN 0 / TN 138 | 0.9906 | 525/526，正样本 387/388 | 387/388 | 101/101，0.2324 px | 525/526 | 51.6 ms，p95 54.1 ms |
| ONNX | TP 388 / FP 0 / FN 0 / TN 138 | 0.9896 | 525/526，正样本 387/388 | 387/388 | 101/101，0.2405 px | 525/526 | 156.7 ms，p95 254.5 ms |

val 结果（501 张）：组件 TP 385 / FP 1 / FN 0 / TN 115，IoU 0.9905，状态端到端 500/501，正样本 385/385，中心 98/98，完全正确 500/501。

#### test 状态混淆矩阵

| GT \ Pred | absent | unchecked | verified | processing | error | unknown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| absent | 138 | 0 | 0 | 0 | 0 | 0 |
| unchecked | 0 | 101 | 0 | 0 | 0 | 0 |
| verified | 0 | 0 | 95 | 0 | 1 | 0 |
| processing | 0 | 0 | 0 | 96 | 0 | 0 |
| error | 0 | 0 | 0 | 0 | 95 | 0 |
| unknown | 0 | 0 | 0 | 0 | 0 | 0 |

唯一错误是一个深色主题的合成 `verified` 样本被判为 `error`，置信度约 0.977；组件框定位正确，oracle crop 也是同样错误，属于 state 模型的外观覆盖不足，不是定位或阈值问题。

#### torch / ONNX 一致性（test，526 张）

| 指标 | 结果 |
| --- | --- |
| 组件存在性差异 | 0/526 |
| 状态差异 | 0/526 |
| 组件框 IoU | 0.9976 |
| 复选框中心 L1 误差 | 0.072 px |
| 中心命中失败 | 0 |
| 延迟 | torch 51.1 ms，ONNX CPU 223.3 ms |

#### 复现

```bash
.venv/bin/python -m unittest -v

# torch 评估
.venv/bin/python evaluate.py \
  --dataset datasets/real-v1 --split test \
  --widget-weights runs/widget-v2/weights/best.pt \
  --state-weights runs/state-v2/weights/best.pt \
  --imgsz-widget 1280 --device cuda:0 \
  --json-out reports/test-real-v1.json \
  --overlay-dir overlays/test-real-v1

# ONNX 评估
.venv/bin/python evaluate.py \
  --dataset datasets/real-v1 --split test \
  --backend onnx \
  --widget-weights models/onnx/widget.onnx \
  --state-weights models/onnx/state.onnx

# torch / ONNX 一致性
.venv/bin/python verify_onnx.py \
  --dataset datasets/real-v1 --split test \
  --torch-widget runs/widget-v2/weights/best.pt \
  --torch-state runs/state-v2/weights/best.pt \
  --onnx-widget models/onnx/widget.onnx \
  --onnx-state models/onnx/state.onnx \
  --manifest models/onnx/manifest.json
```

`datasets/real-v1`、`reports/`、`overlays/` 均为本地评估产物，不提交到仓库。


## 许可证

本项目采用 **GNU Affero General Public License v3.0**，见 [`LICENSE`](LICENSE)。

第三方组件、训练/导出依赖以及由 Ultralytics 训练并导出的 ONNX 模型说明，见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

Ultralytics YOLO 默认使用 AGPL-3.0，其官方说明也把训练模型纳入 AGPL 范围。商业、闭源或 SaaS 部署前，请确认 AGPL 义务或获取 Ultralytics Enterprise License。
