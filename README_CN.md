# TV-LoRA

[English](README.md) | [简体中文](README_CN.md)

论文 **Interleaved Score-Based Diffusion and Structure-Constrained Optimization for Sparse-View CT Reconstruction** 的代码实现。

TV-LoRA 将预训练二维得分扩散模型与 ADMM 更新交错执行，通过投影保真度、三维各向异性全变差和逐切片奇异值收缩实现稀疏视图 CT 重建。本仓库提供 LDCT、CTHD 数据在 2、4、8 个投影视角下的重建、评估与可视化工具。

## 1. 安装环境

克隆仓库并创建环境：

```bash
git clone https://github.com/DENGZY8/TV-LoRA.git
cd TV-LoRA
conda env create -f environment.yml
conda activate tv-lora
```

环境配置采用 Python 3.9 和 PyTorch 2.8.0。使用 GPU 重建时，请安装与 NVIDIA 驱动兼容的 PyTorch CUDA 版本，安装命令参见 [PyTorch 官方文档](https://pytorch.org/get-started/previous-versions/#v280)。检查当前环境：

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

默认 `native` 后端使用 PyTorch 算子，可在 CPU 或 CUDA 上运行，无需编译扩展。可选的 `cuda` 后端需要兼容的 CUDA 工具链、C++ 编译器及 `ninja`。

## 2. 下载预训练模型

LDCT 和 CTHD 共用同一份冻结的 NCSN++ 得分模型。重建时无需重新训练或微调。

将 [checkpoint_185.pth](https://github.com/DENGZY8/TV-LoRA/releases/download/v1.0.0/checkpoint_185.pth) 下载到 `checkpoints/`：

```bash
mkdir -p checkpoints
curl --fail --location --retry 3 \
  --output checkpoints/checkpoint_185.pth \
  https://github.com/DENGZY8/TV-LoRA/releases/download/v1.0.0/checkpoint_185.pth
```

文件大小为 **983,643,489 字节**，SHA256 为：

```text
88f6a5b48dd8c0a21ebe948e15b693b6fa4e33e21b04d09ad12079efb77486de
```

在 CPU 上检查本地文件及模型、指数移动平均（EMA）参数：

```bash
python tools/check_checkpoint.py \
  --checkpoint checkpoints/checkpoint_185.pth \
  --output checkpoint_info.json
```

实验配置会在加载时核对上述 SHA256。权重存放在其他位置时，可通过 `--checkpoint` 指定其准确路径。

## 3. 准备数据

数据集获取页面：

- **LDCT：** [AAPM 低剂量 CT 挑战赛数据集](https://www.aapm.org/GrandChallenge/LowDoseCT/)。
- **CTHD：** [CT Heart Dataset](https://www.kaggle.com/datasets/nikhilroxtomar/ct-heart-segmentation)。

请遵守数据提供方的访问和使用条件。输入应为已预处理的浮点 CT 数组，数值范围为 **[0, 1]**，每张轴向切片大小为 **256 × 256**。每个体积支持以下两种存储方式：

- 一个目录内保存逐切片 `.npy` 文件，文件名体现数值顺序，例如 `000.npy`、`001.npy` 等。
- 单个 `.npy` 体积文件，形状为 `[S, 256, 256]` 或 `[S, 1, 256, 256]`，其中 `S` 为切片数量。

保留解剖切片顺序，对所有比较方法使用一致的预处理。数据加载器接收预处理后的数组，不负责 DICOM 转换或图像缩放。

提供的实验配置使用以下完整体积：

| 数据集 | 切片数 | 示例数据目录 | 实验配置 |
| --- | ---: | --- | --- |
| LDCT | 198 | `data/LDCT/` | `configs/experiments/ldct.json` |
| CTHD | 112 | `data/CTHD/` | `configs/experiments/cthd.json` |

以下命令显式指定示例数据目录，使用时请替换为实际预处理数据的路径。所有命令均在仓库根目录执行。

## 4. 运行重建

重建前检查输入、权重、得分模型前向计算及投影算子：

```bash
python run.py --config configs/experiments/ldct.json \
  --data-dir data/LDCT --device cuda:0 --doctor

python run.py --config configs/experiments/cthd.json \
  --data-dir data/CTHD --device cuda:0 --doctor
```

依次运行 8、4、2 个投影视角：

```bash
python run.py --config configs/experiments/ldct.json \
  --data-dir data/LDCT --device cuda:0 --views 8 4 2

python run.py --config configs/experiments/cthd.json \
  --data-dir data/CTHD --device cuda:0 --views 8 4 2
```

每个配置采用 2,000 个反向扩散步骤、反向扩散预测器、SNR 为 0.16 的单步 Langevin 校正器；每个反向步骤执行一次 ADMM 修正，其中包含一次 CG 递推。名义切片批大小为 12，TV 与谱项权重分别为 0.04 和 0.5。

| 投影视角数 | TV 惩罚参数 | 谱项惩罚参数 |
| ---: | ---: | ---: |
| 8 | 15 | 1 |
| 4 | 15 | 5 |
| 2 | 15 | 1 |

只运行单个视角时，可使用 `--views 8`。其他选项包括 `--checkpoint`、`--output-dir`、`--batch-size`、`--seed` 和 `--op-backend`。完整选项见 `python run.py --help`。

CTHD 定性可视化配置单独保存在 `configs/experiments/cthd_qualitative.json`，采用 TV 惩罚参数 10；8、4、2 个视角的谱项惩罚参数依次为 1、3、3，输出保存在 `outputs/CTHD_qualitative/`。

使用该配置重建并绘图：

```bash
python run.py --config configs/experiments/cthd_qualitative.json \
  --data-dir data/CTHD --device cuda:0 --views 8 4 2
python tools/plot_reconstruction.py --config configs/visualization/cthd_qualitative.json --output figures/CTHD_qualitative
```

### 使用 tmux

安装 `tmux` 并激活环境后执行：

```bash
bash scripts/tmux_run.sh configs/experiments/ldct.json 0 --data-dir data/LDCT
```

LDCT 任务完成后，在同一张 GPU 上启动 CTHD：

```bash
bash scripts/tmux_run.sh configs/experiments/cthd.json 0 --data-dir data/CTHD
```

第二个参数指定物理 GPU 编号。若有两张 GPU 并需要同时运行，可将 CTHD 命令中的编号改为 `1`。启动脚本会打印会话名称和日志路径。

```bash
tmux ls
tmux attach -t SESSION_NAME
```

先按 **Ctrl+b**，再按 **d**，即可脱离会话并保持任务运行。任务结束后，日志保留在 `logs/`。

重新启动时可添加 `--skip-completed`，跳过配置及输出哈希匹配的已完成视角。中断的视角从第一个扩散步骤重新运行。需要替换现有结果时才使用 `--overwrite`。

## 5. 输出与评估

结果分别保存在 `outputs/LDCT/viewN/` 和 `outputs/CTHD/viewN/`，其中 `N` 为 2、4 或 8。

| 文件 | 内容 |
| --- | --- |
| `reconstruction.raw.npy` | 导出缩放前的重建体积 |
| `reconstruction.npy` | 按整卷 min-max 缩放后的重建体积 |
| `reference.npy` | 独立按整卷 min-max 缩放后的参考体积 |
| `measurement.npy` | 模拟的稀疏视图投影 |
| `metrics.json` | 导出后的重建体积与参考体积指标 |
| `metrics.raw.json` | 导出缩放前的指标 |
| `run.json` | 运行环境、耗时、权重标识及完成状态 |
| `config.resolved.json` | 当前视角实际使用的参数 |

每个数据集的输出目录还包含 `results.csv`。PSNR、SSIM 和 Dice 按截面计算，并分别在轴向、冠状面和矢状面内取平均。PSNR、SSIM 的数据范围为 1.0，SSIM 使用均匀加权的 7 × 7 窗口。Dice 将强度大于 0.5 的体素作为前景；两个掩膜均为空时记为零。

单独评估已保存的数组：

```bash
python tools/evaluate.py \
  --reference outputs/LDCT/view8/reference.npy \
  --prediction outputs/LDCT/view8/reconstruction.npy \
  --output outputs/LDCT/view8/metrics_recomputed.json
```

## 6. 可视化

生成重建图：三行依次为轴向、冠状面和矢状面，四列依次为 8 视角、4 视角、2 视角及参考图像。

```bash
python tools/plot_reconstruction.py --config configs/visualization/ldct.json --output figures/LDCT
python tools/plot_reconstruction.py --config configs/visualization/cthd.json --output figures/CTHD
```

图中显示视角数、截面 PSNR/SSIM 及局部放大区域。可在可视化 JSON 中设置 `indices` 和 `roi`，选择截面与感兴趣区域。使用其他重建输出目录时，同时更新配置中的 `reference` 与 `reconstructions` 路径。

生成绝对误差图，以及 LDCT、CTHD 的 PSNR 和 SSIM 对比曲线：

```bash
python tools/plot_error_maps.py --config configs/visualization/error_ldct.json --output figures/LDCT_errors
python tools/plot_error_maps.py --config configs/visualization/error_cthd.json --output figures/CTHD_errors
python tools/plot_curves.py --csv outputs/LDCT/results.csv outputs/CTHD/results.csv --output figures/LDCT_CTHD_curves
```

加入基线误差图时，在 `configs/visualization/error_baselines.example.json` 中填写匹配的重建数组路径，并将该配置传入 `plot_error_maps.py`。所有方法须使用相同的参考体积、切片顺序和强度尺度。

加入基线曲线时，提供包含 `dataset,method,views,plane,psnr,ssim,dice` 列的 CSV 文件。每种方法在其涉及的数据集中均需提供 2、4、8 视角及三个方向的结果；每个曲线点为三个方向均值的等权平均。

图像以 PNG、PDF、SVG 格式导出，同时生成记录源数组和绘图配置的 JSON 文件。

## 7. 测试

运行单元测试与合成数据集成测试：

```bash
python -m pip install pytest
python -m pytest -q tests
```

## 8. 许可证与致谢

参见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。得分模型、SDE、采样及 CT 组件基于 [DiffusionMBIR](https://github.com/hyungjin-chung/DiffusionMBIR) 和 [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch)。
