# simple_nerf
This is a learning-oriented project designed to reproduce the NERF paper in the simplest, and most comprehensible way possible.

# 环境要求
- 本项目采用uv来管理Python的库，因此需要先安装uv环境。
- 本项目使用的是PyTorch的cuda版本，因此需要N卡并保证完成了驱动的安装。

# 程序运行
完成uv环境安装之后，可通过如下的简单步骤运行本项目。
```bash
git clone https://github.com/weixinhum2023/simple_nerf
cd simple_nerf\
uv sync
```
因为uv在Windows上的PyTorch包是不带GPU的版本，因此需要自行安装下PyTorch，这里直接下载最新版本(2.10.0)
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
随后使用安装好的环境运行程序即可：
```bash
uv run python train.py
```


# TensorBoard 可视化
训练脚本会自动将 TensorBoard 日志写入 `logs/tensorboard`，包含：

- 训练标量：`loss_coarse`、`loss_fine`、`loss_total`、`psnr_coarse`、`psnr_fine`、`lr`、`step_time`
- 测试渲染图：`test_render/coarse`、`test_render/fine`

启动方式：

```bash
tensorboard --logdir ./logs/tensorboard --port 6006
```
或添加局域网访问
```bash
tensorboard --logdir ./logs/tensorboard --port 6006 --host=192.168.31.237
```

浏览器访问：`http://localhost:6006`
