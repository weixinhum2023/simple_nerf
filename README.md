# simple_nerf
This is a learning-oriented project designed to reproduce the NERF paper in the simplest, most comprehensible, and fastest way possible.

## TensorBoard 可视化

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
