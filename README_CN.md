# TV-LoRA 运行说明

本包整理当前论文的 LDCT、CTHD 重建与评估流程，保留原有 TV-LoRA 核心更新函数。英文 README 包含完整参数、指标和图像处理定义。

## 1. 环境和权重

复用服务器当前可运行环境：

```bash
conda activate mypy
cd /path/to/TV-LoRA
```

模型结构位于 `models/`。预训练权重可保留在原目录，通过 `--checkpoint` 指定文件，无需为两个数据集各复制一份。

```bash
python tools/check_checkpoint.py \
  --checkpoint /path/to/checkpoint_185 \
  --output checkpoint_info.json
```

如果实际文件带 `.pth`，请使用该准确文件名。此命令在 CPU 上核对模型键、EMA 数量与形状，输出权重 SHA256 和大小。不要通过重命名来替代文件有效性检查。

## 2. 长任务前检查

```bash
python run.py --config configs/experiments/ldct.json \
  --data-dir /path/to/LDCT/slices \
  --checkpoint /path/to/valid/checkpoint_185 --device cuda:0 --doctor
```

LDCT 默认核对 198 张切片，CTHD 核对 112 张。完整体积保留，不截断；实际数据、权重和输出目录均可通过命令行覆盖。模型加载失败会停止运行。

## 3. tmux 重建

```bash
bash scripts/tmux_run.sh configs/experiments/ldct.json 0 \
  --data-dir /path/to/LDCT/slices \
  --checkpoint /path/to/valid/checkpoint_185

bash scripts/tmux_run.sh configs/experiments/cthd.json 1 \
  --data-dir /path/to/CTHD/slices \
  --checkpoint /path/to/valid/checkpoint_185
```

第二个参数是物理 GPU 编号；脚本会打印会话名和日志路径。`tmux attach -t 会话名` 查看，Ctrl+b 后按 d 脱离。一次依次运行 8、4、2 views。完成后会话关闭，日志保留。

主表配置统一为：8 views 使用 `(15,1)`，4 views 使用 `(15,5)`，2 views 使用 `(15,1)`；前后分别为 TV 和谱项 ADMM 惩罚。显式正则权重为 0.04、0.5。

`cthd_legacy_visual.json` 单独保留旧 CTHD 可视化配置：`(10,1)`、`(10,3)`、`(10,3)`，输出目录独立。

`--skip-completed` 只跳过配置和文件哈希均匹配的已完成视角；中断的扩散轨迹需要从头重建。

## 4. 评估和绘图

重建自动输出原始数组、与旧导出处理一致的整卷 min-max 数组、PSNR/SSIM/Dice、参数、输入顺序、哈希和计时记录。

```bash
python tools/plot_reconstruction.py --config configs/visualization/ldct.json --output figures/LDCT
python tools/plot_reconstruction.py --config configs/visualization/cthd.json --output figures/CTHD
python tools/plot_error_maps.py --config configs/visualization/error_ldct.json --output figures/LDCT_errors
python tools/plot_curves.py --csv outputs/LDCT/results.csv outputs/CTHD/results.csv --output figures/LDCT_CTHD_curves
```

可视化默认取中心截面，支持在配置中修改截面与黄色 ROI，显示数据来自实际输入数组。误差图加入其他方法时，将配置中的方法名映射到对应的 `.npy` 文件。论文旧 ADMM-TV/FBPConvNet 图块的绘制路径见 `docs/paper_fig5.md`。

## 5. 上传 GitHub

将本目录内容放在 `DENGZY8/TV-LoRA` 仓库根目录。大权重可放在仓库 Releases，README 中填写准确下载链接与 SHA256。`.gitignore` 已排除数据、权重、输出、日志和缓存。

完整 GPU 重建尚需在原服务器上使用原数据、原权重验证；本包不附带原始 CT 和权重。
