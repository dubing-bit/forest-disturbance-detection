# Forest Disturbance Detection

[English](README.md) | [简体中文](README.zh-CN.md)

## 代码来源

本仓库提供 Du 等人论文 *A Combined Deep Learning and Prior Knowledge Constraint Approach for Large-Scale Forest Disturbance Detection Using Time Series Remote Sensing Data* 的研究代码，使用时间序列 Transformer 与先验知识约束检测森林扰动年份。

论文发表于 *Remote Sensing*，2023，15(12)，2963；DOI：[10.3390/rs15122963](https://doi.org/10.3390/rs15122963)。仓库随附 NBR 指数的 W7、W9 和 W11 模型权重，说明见 [models/README.md](models/README.md)。原始研究区影像、训练样本和人工验证标签未随仓库提供。

## 安装

需要 Python 3.10 或更高版本、Git，以及 CPU 或支持 CUDA 的 GPU。

```bash
git clone https://github.com/dubing-bit/forest-disturbance-detection.git
cd forest-disturbance-detection
```

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
```

Linux/macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

如需使用 CUDA，请先按照 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)安装适合当前硬件与驱动的 PyTorch，再安装本项目。

## 运行示例

安装后可使用 `forest-disturbance`，也可使用等价命令 `python -m forest_disturbance`。命令向终端输出 JSON 格式的运行摘要。

### 合成示例

在仓库根目录运行：

```bash
python scripts/make_demo.py
forest-disturbance predict --input outputs/demo/annual_nbr.tif --output outputs/demo/disturbance_year.tif --checkpoint models/PET_NBR_W9.pth --start-year 2000
```

脚本生成合成 NBR 栅格 `outputs/demo/annual_nbr.tif` 和训练窗口 `outputs/demo/training_windows.csv`。合成数据用于演示安装与运行流程，不用于评价真实影像检测精度。

### 真实影像预测

输入为按年份从早到晚排列的单指数多波段 GeoTIFF，每个波段表示连续一年的 NBR。例如，2000–2020 年应对应 21 个波段。输入必须具有 CRS、仿射变换以及正确的 NoData 值或有效掩膜。云掩膜、年度合成、指数计算、时间插值与 SG 滤波等预处理需按需事先完成。

下面使用 W11 权重、时间窗口步长 4 和存储值缩放因子 10000：

```bash
forest-disturbance predict --input data/annual_nbr_2000_2020.tif --output outputs/disturbance_year.tif --checkpoint models/PET_NBR_W11.pth --start-year 2000 --step 4 --scale-factor 10000 --min-slope -0.05 --block-size 128 --batch-size 4096 --device auto
```

一个像元只要任意一年缺测，就跳过其整个时间序列。可通过 `--mask data/forest_mask.tif` 指定单波段森林掩膜；掩膜须与输入具有相同 CRS、尺寸和仿射变换，仅有效且大于 0 的区域参与检测。未声明为 NoData 的 0 不会自动被视为缺测。

输出为保留输入 CRS、分辨率和仿射变换的单波段压缩 `int16` GeoTIFF：

| 输出值 | 含义 |
| --- | --- |
| 正整数，例如 `2009` | 最早接受的扰动公历年份 |
| `0` | 有效像元未检出符合条件的扰动 |
| `-32768` | NoData、任一年缺测或被森林掩膜排除 |

输出已经是公历年份，无需再加起始年份。运行时如需覆盖已有文件，添加 `--overwrite`。

### 训练

训练输入为固定长度时间窗口 CSV，每行一个样本。时间特征列按从早到晚排列；`label` 为 `0`（稳定）或 `1`（窗口内包含扰动）。W9 的格式示例如下：

```csv
t0,t1,t2,t3,t4,t5,t6,t7,t8,label
0.65,0.66,0.64,0.65,0.66,0.65,0.64,0.65,0.66,0
0.64,0.65,0.66,0.64,0.65,0.66,0.65,0.64,0.65,0
0.65,0.66,0.64,0.65,0.20,0.21,0.22,0.23,0.24,1
0.64,0.65,0.66,0.64,0.19,0.20,0.21,0.22,0.23,1
```

每类至少需要两个样本。除标签外的所有列默认均作为特征；若 CSV 含 ID、坐标等列，应通过 `--feature-columns` 显式选择时间列。年度扰动标签需事先转换为窗口二分类标签。

使用合成示例生成的训练 CSV：

```bash
forest-disturbance train --csv outputs/demo/training_windows.csv --checkpoint outputs/model.pth --window-size 9 --feature-columns t0 t1 t2 t3 t4 t5 t6 t7 t8 --scale-factor 1 --epochs 100 --patience 20 --seed 42 --device auto
```

训练按验证集损失保存最佳权重。权重包含模型结构、特征列与缩放因子等元数据；预测默认读取其中的缩放因子。因此，训练 CSV 与预测栅格应使用相同的存储单位；单位不同时，需显式指定预测的 `--scale-factor`。例如，已归一化指数使用 `1`，放大 10000 倍存储的指数使用 `10000`。

### 年份精度评价

准备包含真实值与预测值的 CSV：

```csv
reference,prediction
0,0
2009,2009
2012,2013
2015,0
```

`0` 表示无扰动，正整数表示公历年份。评价前应排除 NoData；不能将 `-32768` 作为无扰动值。

```bash
forest-disturbance evaluate --csv data/validation.csv --tolerance 0
```

命令输出年份精度、二分类精度、扰动精确率、召回率、F1、漏检率与误检率等指标。`--tolerance 0` 要求年份完全一致；`--tolerance 1` 允许相差一年。超出容差的扰动同时计入漏检和误检；分母为 0 的指标返回 `null`。

## 部署示例

下面以 Linux 服务器为例，将仓库部署到 `/srv/forest-disturbance-detection`，输入放在 `/srv/data`，结果写入 `/srv/results`。请按实际目录替换路径。

```bash
git clone https://github.com/dubing-bit/forest-disturbance-detection.git /srv/forest-disturbance-detection
python3 -m venv /srv/forest-disturbance-detection/.venv
/srv/forest-disturbance-detection/.venv/bin/python -m pip install /srv/forest-disturbance-detection
mkdir -p /srv/results

/srv/forest-disturbance-detection/.venv/bin/python -m forest_disturbance predict \
  --input /srv/data/annual_nbr_2000_2020.tif \
  --output /srv/results/disturbance_year.tif \
  --checkpoint /srv/forest-disturbance-detection/models/PET_NBR_W11.pth \
  --start-year 2000 \
  --step 4 \
  --scale-factor 10000 \
  --block-size 128 \
  --batch-size 4096 \
  --device auto
```

服务器使用 CUDA 时，在安装项目之前，先向上述虚拟环境安装匹配环境的 PyTorch。显存不足时减小 `--batch-size`；内存不足时减小 `--block-size`。

## 参数说明

### `predict`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--input` | 必填 | 按年排列的单指数多波段 GeoTIFF 路径 |
| `--output` | 必填 | 扰动年份 GeoTIFF 输出路径 |
| `--checkpoint` | 必填 | 模型权重路径；随附 W7、W9、W11 NBR 权重 |
| `--start-year` | 必填 | 第一波段对应的公历年份 |
| `--window-size` | 从权重推断 | 时间窗口长度，显式指定时须与权重一致 |
| `--step` | `1` | 时间窗口步长；论文 W7、W9、W11 设置分别为 2、2、4 |
| `--batch-size` | `4096` | 每次模型推理的窗口数 |
| `--block-size` | `128` | 读取空间块的边长，单位为像元 |
| `--scale-factor` | 权重元数据值，否则为 `10000` | 模型输入等于存储值除以该因子；须为正数 |
| `--min-slope` | `-0.05` | 趋势斜率须严格小于该阈值 |
| `--min-intercept` | `0.2` | 拟合截距须严格大于该阈值 |
| `--min-drop` | `-0.1` | 候选相邻年度差值须严格小于该阈值 |
| `--padding` | `legacy` | `legacy`：源程序补边布局；`edge`：重复端点值；`linear`：每个补边位置按 0.001 延伸端点值 |
| `--mask` | 无 | 可选单波段森林掩膜路径 |
| `--overwrite` | 关闭 | 允许覆盖已有输出文件 |

### `train`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--csv` | 必填 | 训练窗口 CSV 路径 |
| `--checkpoint` | 必填 | 最佳模型权重输出路径 |
| `--window-size` | `9` | 时间窗口长度，可选 `7`、`9`、`11` |
| `--feature-columns` | 标签列以外的所有列 | 按时间顺序指定一个或多个特征列名，列名间用空格分隔 |
| `--label-column` | `label` | 二分类标签列名 |
| `--scale-factor` | `1.0` | 训练特征除以该因子后送入模型；须为正数 |
| `--epochs` | `100` | 最大训练轮数 |
| `--batch-size` | `64` | 训练批次大小 |
| `--learning-rate` | `0.001` | Adam 优化器学习率 |
| `--validation-fraction` | `0.2` | 验证集比例，须大于 0 且小于 1 |
| `--patience` | `20` | 验证损失连续未改善多少轮后停止训练 |
| `--seed` | `42` | 数据划分与训练随机种子 |
| `--no-balance` | 关闭 | 禁用训练集类别平衡采样 |
| `--overwrite` | 关闭 | 允许覆盖已有权重文件 |

以下参数同时适用于 `predict` 和 `train`：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--device` | `auto` | `auto` 在 CUDA 可用时使用 GPU，否则使用 CPU；也可指定 `cpu`、`cuda`、`cuda:0` 等 |
| `--threads` | CPU 数量与 `4` 的较小值 | PyTorch CPU 线程数，须为正整数 |

### `evaluate`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--csv` | 必填 | 包含真实年份与预测年份的 CSV 路径 |
| `--reference-column` | `reference` | 真实年份列名 |
| `--prediction-column` | `prediction` | 预测年份列名 |
| `--tolerance` | `0` | 允许的年份绝对差，须为非负整数 |

所有命令均支持 `-h` 或 `--help`，例如 `forest-disturbance predict --help`。

## 参考文献

Du, B.; Yuan, Z.; Bo, Y.; Zhang, Y. A Combined Deep Learning and Prior Knowledge Constraint Approach for Large-Scale Forest Disturbance Detection Using Time Series Remote Sensing Data. *Remote Sensing* **2023**, *15*(12), 2963. [https://doi.org/10.3390/rs15122963](https://doi.org/10.3390/rs15122963).

## 引用与许可

研究中使用本代码时，请引用相关论文：

```bibtex
@article{du2023forest,
  title = {A Combined Deep Learning and Prior Knowledge Constraint Approach for Large-Scale Forest Disturbance Detection Using Time Series Remote Sensing Data},
  author = {Du, Bing and Yuan, Zhanliang and Bo, Yanchen and Zhang, Yusha},
  journal = {Remote Sensing},
  volume = {15},
  number = {12},
  pages = {2963},
  year = {2023},
  doi = {10.3390/rs15122963}
}
```

机器可读引用信息见 [CITATION.cff](CITATION.cff)。代码按 [Apache License 2.0](LICENSE) 发布；版权与来源声明见 [NOTICE](NOTICE)。随附权重的来源和使用说明见 [models/README.md](models/README.md)。
