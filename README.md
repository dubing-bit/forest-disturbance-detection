# Forest Disturbance Detection

基于时间序列 Transformer 与先验知识约束的森林扰动年份检测。
本仓库整理自 Du 等人的 [Remote Sensing 2023 论文源代码](https://doi.org/10.3390/rs15122963)，
重点优化大幅遥感影像预测：**分块读取 → 时间窗口批量推理 → 向量化扰动定位 → 分块写出**。

程序仍对所有有效像元进行检测，但不再用 Python 逐个像元调用趋势拟合。
影像大小不再决定整幅时间窗口张量的内存占用；`--block-size` 控制空间块，
`--batch-size` 控制一次送入模型的窗口数，支持 CPU 和 CUDA。

> 本发布版完成软件重构与一致性验证。原始研究区影像、训练样本和人工验证标签未随源码提供，
> 因而不声称复现论文的 87.8% 精度。随附权重需要在你的研究区独立验证。

## 安装

需要 Python 3.10+。Windows PowerShell：

```powershell
git clone https://github.com/dubing-bit/forest-disturbance-detection.git
cd forest-disturbance-detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Linux/macOS 将环境激活命令替换为 `source .venv/bin/activate`。
如果需要 GPU，请按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)
选择匹配驱动的 CUDA 版本，然后安装本项目。`--device auto` 在 CUDA 可用时自动使用 GPU。

## 快速运行

下面生成一份**合成** NBR 时间栈，用于检查安装和读写流程：

```bash
python scripts/make_demo.py
forest-disturbance predict --input outputs/demo/annual_nbr.tif --output outputs/demo/disturbance_year.tif --checkpoint models/PET_NBR_W9.pth --start-year 2000
```

也可以用 `python -m forest_disturbance` 替代 `forest-disturbance`。
合成示例没有科学验证意义；真实影像请遵循下列输入规范。

## 实际影像预测

输入为**按年份从早到晚排列的单指数多波段 GeoTIFF**，每个波段代表连续一年的 NBR。
例如 2000–2020 年是 21 个波段；不是单景影像的多个光谱波段。
输入必须有 CRS、仿射变换和正确的缺测值/有效掩膜。
程序不负责 Landsat 云掩膜、年度合成、NBR 计算、时间插值或 SG 滤波。

论文最佳窗口设置的调用示例（W11、步长4、S-DRI阈值−0.05）：

```bash
forest-disturbance predict --input data/annual_nbr_2000_2020.tif --output outputs/disturbance_year.tif --checkpoint models/PET_NBR_W11.pth --start-year 2000 --step 4 --min-slope -0.05 --scale-factor 10000 --block-size 128 --batch-size 4096 --device auto
```

参数说明：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--checkpoint` | 必填 | 随附 W7/W9/W11 权重，或新训练的元数据权重 |
| `--start-year` | 必填 | 第一波段对应的实际年份；不从文件名猜测 |
| `--window-size` | 从权重推断 | 手动指定时必须与权重一致 |
| `--step` | 1 | 时间窗口步长；论文 W7/W9/W11 分别采用2/2/4 |
| `--scale-factor` | 元数据值，否则10000 | 模型输入 = 存储值 / 此数值；已归一化指数使用1 |
| `--block-size` | 128 | 空间块边长，限制影像读取内存 |
| `--batch-size` | 4096 | 每次推理的时间窗口数，限制模型内存 |
| `--threads` | 最多4 | PyTorch CPU线程数；按机器实测调整 |
| `--min-slope` | −0.05 | 趋势斜率严格小于此阈值 |
| `--min-intercept` | 0.2 | 拟合截距严格大于此阈值 |
| `--min-drop` | −0.1 | 候选相邻年度差值严格小于此阈值 |
| `--padding` | `legacy` | 保留原影像程序补边布局；还可选 `edge`、`linear` |
| `--mask` | 无 | 可选单波段森林掩膜；与输入完全同网格、同CRS，有效正值参与检测 |
| `--overwrite` | 关闭 | 显式允许覆盖已有结果 |

一个像元只要任意一年缺测，就跳过整个时间序列；非森林掩膜区也跳过。
未声明为 NoData 的0不会自动当作缺测。运行前请检查输入有效性和数值尺度。
若显存不足，先减小 `--batch-size`；内存不足则减小 `--block-size`。

输出为保留原 CRS、分辨率和仿射变换的单波段、压缩 `int16` GeoTIFF：

| 值 | 意义 |
| --- | --- |
| 正整数，如2009 | 最早接受的扰动**公历年份** |
| 0 | 有效像元中未检出满足条件的扰动 |
| −32768 | NoData、缺年或被森林掩膜排除 |

结果不需要再加2000。多个窗口命中时保留最早年份。
新版修复了旧代码“首年索引0与无扰动混用”和越界索引问题。
论文评估范围是2001–2020；若比较论文结果，应使用同样年份范围和独立标签。

## 优化方式与性能验证

| 原实现 | 当前实现 |
| --- | --- |
| 整幅读取并构造所有滑窗 | Rasterio空间分块，只保留当前块 |
| 每个时间窗口逐行调用模型 | 跨像元、跨时间窗口组成固定上限批次 |
| 候选像元逐个排序和 `np.polyfit` | NumPy批量筛选和四点最小二乘闭式计算 |
| 全部窗口都调用神经网络 | 先检查同一套必要规则，仅对通过规则的窗口调用模型 |
| 保留多年份扰动结果立方体 | 直接更新每个像元最早年份 |
| 重复 softmax | log概率直接 `argmax` |

可重复运行的基准使用**真实随附权重 + 合成时间栈**，对比原有逐行模型推理/逐像元规则与新版批量路径，
计时前预热，并断言输出一致；另独立对比标量和向量规则。
测试范围是内存中的 CPU 预测，不包括磁盘读取和权重加载；不会代表所有影像或 GPU 的加速倍数。

```bash
python scripts/benchmark.py --checkpoint models/PET_NBR_W9.pth --threads 4 --repeats 5
```

已测环境：i5-13500HX、Windows、CPU 4线程，21年 × 24 × 32像元，重复5次取中位数。
该合成任务由1.009秒降至0.0595秒，约 **17倍**；模型调用由504次降至4次，输出完全一致。
规则内核的独立测试约快399倍。这些数字仅描述本次工作负载，详见 [基准记录](docs/benchmark.json)。

## 训练

训练输入为已提取的固定长度时间窗口 CSV，每行一个样本，`label` 为0（稳定）或1（窗口含扰动）。
W9示例表头：`t0,t1,t2,t3,t4,t5,t6,t7,t8,label`。
其他列默认都会被当作特征；含ID或经纬度时必须用 `--feature-columns` 明确选择时间列。
年度标签和整段时间序列需要先转换为窗口样本；程序不自动解释“扰动年份”标签。

```bash
forest-disturbance train --csv outputs/demo/training_windows.csv --checkpoint outputs/model.pth --window-size 9 --scale-factor 1 --epochs 100 --patience 20 --seed 42
```

训练只对训练集做类别平衡；验证集不重采样、不舍入概率，按最佳验证损失保存权重并早停。
新权重包含模型结构、特征列、缩放因子和验证指标，CLI预测默认读取其中的缩放因子。
这要求训练CSV与待预测栅格使用相同的存储单位；如果单位不同，应显式覆盖预测的 `--scale-factor`。
内置分层随机划分用于训练监控；正式论文评价仍需独立的空间/时间测试集，避免窗口泄漏。
完整参数见 `forest-disturbance train --help`。

## 年份精度评价

准备含 `reference,prediction` 两列的 CSV，0代表稳定，正数为公历年份。
请先显式排除 NoData，不能把−32768当成无扰动。

```bash
forest-disturbance evaluate --csv data/validation.csv --tolerance 0
```

输出同年精度、二分类精度、扰动精确率/召回率/F1、漏检率和误检率。
年份超出容差的扰动同时计入漏检和误检；分母为0的指标返回 `null`。
`--tolerance 1` 允许±1年，但不能与严格同年评价直接比较。

## Python API

```python
from forest_disturbance.inference import PredictConfig, predict_raster
from forest_disturbance.model import load_model

model = load_model("models/PET_NBR_W9.pth", device="cpu")
config = PredictConfig(window_size=9, start_year=2000, scale_factor=10000)
summary = predict_raster("data/annual_nbr.tif", "outputs/year.tif", model, config)
print(summary)
```

`predict_series(model, series, config)` 也接受 `(像元数, 年数)` NumPy数组。
GeoTIFF场景推荐使用 `predict_raster` 来控制空间内存。

## 目录与开发

```text
src/forest_disturbance/  模型、数据、训练、规则、批量预测、评价和CLI
models/                 三种原始NBR权重及SHA-256记录
scripts/                合成示例和可重复性能基准
tests/                  独立参考一致性、模型、训练与GeoTIFF回归测试
docs/                   方法约定、迁移说明和性能记录
.github/workflows/      Windows/Linux持续集成
CITATION.cff            论文引用元数据
LICENSE                 Apache-2.0
```

```bash
ruff check src tests scripts
pytest -q
python -m build
```

测试包含旧版模型计算一致性、批次独立性、三种权重加载、向量规则与标量 `np.polyfit` 一致性、
分块边界、首年检测、NoData/掩膜、输出地理信息、失败时避免留下半成品，以及训练早停。
详见 [贡献指南](CONTRIBUTING.md)、[方法与边界约定](docs/method.md)、[旧代码迁移说明](docs/migration.md)。

## 引用与许可证

```bibtex
@article{du2023forest,
  title={A Combined Deep Learning and Prior Knowledge Constraint Approach for Large-Scale Forest Disturbance Detection Using Time Series Remote Sensing Data},
  author={Du, Bing and Yuan, Zhanliang and Bo, Yanchen and Zhang, Yusha},
  journal={Remote Sensing}, volume={15}, number={12}, pages={2963}, year={2023},
  doi={10.3390/rs15122963}
}
```

代码按 [Apache-2.0](LICENSE) 许可发布。论文引用信息见 [CITATION.cff](CITATION.cff)；
随附权重来源、校验值和使用限制见 [models/README.md](models/README.md)。
