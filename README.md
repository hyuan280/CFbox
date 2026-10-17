# CFbox

使用本地 HTML/CSS 模拟验证组件，通过 **Camofox REST API** 访问页面、截取视口并读取 DOM 边界，自动导出两阶段 YOLO 数据集；另含 CPU 训练、两阶段推理和评估脚本。无需 Playwright 客户端或大模型。

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

默认 CPU、8 个计算线程、batch=2、数据加载 workers=0。全图默认 1280，局部默认 640；可通过 `--threads`、`--batch`、`--imgsz` 调整。权重保存在 `models/`；训练结果保存在 `runs/<stage>/`，重复运行自动使用新目录。`weights/best.pt` 是验证集最优权重，`weights/last.pt` 是最后一轮权重。首次运行可能额外下载绘图字体。只有权重来自可信来源时才加载 `.pt` 文件。

快速检查环境而非训练可用模型：

```bash
.venv/bin/python train.py --stage widget --epochs 1 --imgsz 320 --name smoke-widget
.venv/bin/python train.py --stage state --epochs 1 --imgsz 320 --name smoke-state
```

50 张合成样本训练出的权重只能检查管线，不能据此评价实际识别效果。ONNX 与 ONNX Runtime 在后续模型导出和部署时再安装，本阶段不需要。

## 模型接入

`cfbox_detector.py` 封装两阶段推理，`predict.py` 提供命令行入口。默认读取：

```text
runs/widget/weights/best.pt
runs/state/weights/best.pt
```

单张截图：

```bash
.venv/bin/python predict.py --image datasets/demo/widget/images/test/sample_000041.png --json
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

detector = CFChallengeDetector()
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

## 评估当前模型

```bash
.venv/bin/python evaluate.py \
  --split test \
  --overlay-dir overlays/test \
  --json-out reports/test-metrics.json
```

`--split` 可选 `train`、`val`、`test`、`all`。`all` 和 `train` 包含训练图片，结果不能当作独立测试。脚本计算：

- 组件存在性的 TP/FP/FN/TN、召回率和误报率。
- 组件框 IoU。
- 端到端状态准确率和混淆矩阵。
- 仅用标注裁剪图时的状态准确率，用于区分状态模型和 widget 定位误差。
- 未勾选复选框中心命中率、像素误差和相对对角线误差。
- 端到端完全正确率和 CPU 延迟。

当前 50 轮训练权重的实测结果（合成数据）：

| 数据 | 组件召回 | 无组件误报 | 组件 IoU | 状态 top-1 | 未勾选中心 |
| --- | --- | --- | --- | --- | --- |
| test，5 张 | 4/4 | 0/1 | 0.932 | 2/4 | 1/1，误差 0.42 px |
| val，5 张 | 4/4 | 0/1 | 0.969 | 2/4 | 1/1，误差 0.47 px |
| all，50 张 | 40/40 | 0/10 | 0.948 | 18/40 | 10/10，误差 0.48 px |

组件检测和复选框定位已经可用；**状态识别目前不可用**。典型错误是 `verified`、`processing` 被判为 `unknown`，以及 `processing` 被判为 `unchecked`。测试集只有 5 张，这些比例没有统计意义。

注意状态模型的 YOLO mAP 仍可能很高，因为不同状态框高度重叠，mAP 不能反映互斥状态的 top-1 分类能力。应以状态准确率、混淆矩阵和端到端完全正确率为准。下一步应把数据扩到至少数百张，并覆盖真实截图；若状态排序仍然混淆，建议改用“裁剪图单标签分类 + 复选框检测”的双头方案，而不是继续只调检测阈值。

```bash
python3 -m unittest -v
```

CPU 全图训练可能耗时较长。检查真实截图上的组件召回率、状态混淆和中心点命中率，再决定是否扩大训练。
