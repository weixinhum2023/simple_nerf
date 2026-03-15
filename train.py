import os
import time
from types import SimpleNamespace

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F

from Embedder import positional_encoding
from NerfNetwork import NeRF
from ReadData import load_blender_data
from VolumeRendering import raw2outputs


def mse2psnr(mse):
	return -10.0 * torch.log10(mse)


def run_network(inputs, viewdirs, model, multires, multires_views):
	"""对采样点编码后一次性送入网络。"""
	input_flat = inputs.reshape(-1, inputs.shape[-1])# 将输入的采样点位置编码成一个二维张量，第一维是所有采样点的数量，第二维是位置编码后的维度。这个位置编码的过程是为了让网络能够更好地利用空间信息，增强对场景的理解能力。
	embedded_pts = positional_encoding(input_flat, multires=multires, use_encoding=True)

	viewdirs_expanded = viewdirs[:, None, :].expand(inputs.shape)
	viewdirs_flat = viewdirs_expanded.reshape(-1, viewdirs_expanded.shape[-1])
	embedded_views = positional_encoding(viewdirs_flat, multires=multires_views, use_encoding=True)

	network_input = torch.cat([embedded_pts, embedded_views], dim=-1)
	outputs = model(network_input)
	outputs = outputs.reshape(inputs.shape[0], inputs.shape[1], 4)
	return outputs


def render_rays(
	rays_o,
	rays_d,
	model,
	near,
	far,
	n_samples,
	multires,
	multires_views,
	white_bkgd,
):
	"""仅进行 coarse 采样与体渲染。"""
	n_rays = rays_o.shape[0]
	t_vals = torch.linspace(0.0, 1.0, n_samples, device=rays_o.device)
	z_vals = near * (1.0 - t_vals) + far * t_vals
	z_vals = z_vals.expand(n_rays, n_samples)# expand函数的作用是将z_vals这个一维张量扩展成一个二维张量，第一维的大小是n_rays，第二维的大小是n_samples。每一行都是原来的z_vals，这样每条光线就有了对应的采样点深度值。

	pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
	viewdirs = rays_d # rays_d已经是单位向量了，不需要再进行归一化了。

	raw = run_network(
		pts,
		viewdirs,
		model,
		multires=multires,
		multires_views=multires_views,
	)

	rgb = raw[..., :3]
	sigma = raw[..., 3]
	rgb_map, _, _, _ = raw2outputs(rgb, sigma, z_vals, white_bkgd=white_bkgd)
	return rgb_map


def render_test_image(model, rays_o, rays_d, H, W, args):
	"""按小批量光线渲染整张测试图，避免显存峰值过高。"""
	outputs = []
	with torch.no_grad():
		for start in range(0, rays_o.shape[0], args.test_render_batch):
			end = start + args.test_render_batch
			pred_rgb = render_rays(
				rays_o[start:end],
				rays_d[start:end],
				model,
				near=args.near,
				far=args.far,
				n_samples=args.n_samples,
				multires=args.multires,
				multires_views=args.multires_views,
				white_bkgd=True,
			)
			outputs.append(pred_rgb)
	img = torch.cat(outputs, dim=0).reshape(H, W, 3)
	return torch.clamp(img, 0.0, 1.0)


def train(args):
    # 设置设备，如果有GPU可用则使用GPU，否则使用CPU
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Use device: {device}")
    # 加载数据集，得到图像、相机位姿、图像的高宽、焦距和训练/测试集的索引列表
	H, W, i_split, rays_o_all, rays_d_all, rgbs_all = load_blender_data(args.datadir, testskip=8)
	i_train = i_split[0]
	# 预计算光线一次性搬到GPU，训练阶段不再重复拷贝。
	rays_o_all = rays_o_all.to(device)
	rays_d_all = rays_d_all.to(device)
	rgbs_all = rgbs_all.to(device)
	# 每张图像的光线数量，即每张图像的像素数量
	n_rays_per_image = rays_o_all.shape[1]
    # 创建NeRF模型和优化器，模型移动到设备上
	model = NeRF().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate)
    # 创建日志文件夹和日志文件路径
	os.makedirs(args.basedir, exist_ok=True)
	preview_dir = os.path.join(args.basedir, "test_renders")
	os.makedirs(preview_dir, exist_ok=True)
	log_path = os.path.join(args.basedir, "train_log.txt")

	i_test = i_split[2]
	test_img_i = int(i_test[0]) if len(i_test) > 0 else int(i_train[0])
	test_rays_o = rays_o_all[test_img_i]
	test_rays_d = rays_d_all[test_img_i]
    # 训练循环，迭代次数为args.n_iters，每次迭代随机选取一张图像进行训练
	t0 = time.time()
	for step in range(1, args.n_iters + 1):
		# 随机选取一张图像，并从中随机选取args.n_rand条光线进行训练
		img_i = np.random.choice(i_train)
		select_inds = torch.randint(0, n_rays_per_image, (args.n_rand,), device=device)
		rays_o = rays_o_all[img_i, select_inds]
		rays_d = rays_d_all[img_i, select_inds]
		target_rgb = rgbs_all[img_i, select_inds]
        # 对选取的光线进行渲染，得到预测的RGB颜色值
		pred_rgb = render_rays(
			rays_o,
			rays_d,
			model,
			near=args.near,
			far=args.far,
			n_samples=args.n_samples,
			multires=args.multires,
			multires_views=args.multires_views,
			white_bkgd=True,
		)
        # 计算预测的RGB颜色值与目标RGB颜色值之间的均方误差损失，并计算PSNR值
		loss = F.mse_loss(pred_rgb, target_rgb)
		psnr = mse2psnr(loss.detach())
        # 反向传播和优化
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
        # 学习率衰减，按照指数衰减的方式调整学习率，衰减的速度由args.lrate_decay控制
		new_lrate = args.lrate * (0.1 ** (step / args.lrate_decay))
		for param_group in optimizer.param_groups:
			param_group["lr"] = new_lrate
        # 每隔args.i_print迭代打印一次训练日志，日志内容包括当前迭代次数、损失值、PSNR值、学习率和每次迭代的时间
		if step % args.i_print == 0 or step == 1:
			dt = time.time() - t0
			msg = (
				f"[Step {step:06d}] Loss={loss.item():.6f} PSNR={psnr.item():.2f} "
				f"LR={new_lrate:.6e} Time={dt:.2f}s"
			)
			print(msg)
			with open(log_path, "a", encoding="utf-8") as f:
				f.write(msg + "\n")
			t0 = time.time()

		if step in (1, 5000, 10000, 20000, 50000, 100000, 150000, 200000):
			was_training = model.training
			model.eval()
			test_img = render_test_image(model, test_rays_o, test_rays_d, H, W, args)
			if was_training:
				model.train()

			out_path = os.path.join(preview_dir, f"test_step_{step:06d}.png")
			iio.imwrite(out_path, (test_img.detach().cpu().numpy() * 255.0).astype(np.uint8))
			print(f"Saved test render to {out_path}")
        # 每隔args.i_weights迭代保存一次模型权重，保存的内容包括当前迭代次数、模型的状态字典和优化器的状态字典，保存的文件名包含当前迭代次数
		if step % args.i_weights == 0 or step == args.n_iters:
			ckpt_path = os.path.join(args.basedir, f"coarse_{step:06d}.pth")
			torch.save(
				{
					"global_step": step,
					"model_state_dict": model.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
				},
				ckpt_path,
			)
			print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
	# 训练配置：直接修改这些变量即可
	args = SimpleNamespace(
		datadir="./lego",
		basedir="./logs",
		n_iters=200001,  # 训练迭代次数，一次迭代中会随机选取一张图像进行训练，迭代次数越多，模型的拟合能力越强，但训练时间也会更长。
		n_rand=1024,  # 每次迭代中随机选取的光线数量，通常是1024或2048。这个值越大，每次迭代的训练效果越好，但显存占用也会更高，训练时间也会更长。
		n_samples=64,  # 每条光线上采样的点的数量，通常是64或128。这个值越大，模型的拟合能力越强，但显存占用也会更高，训练时间也会更长。
		near=2.0,  # 近裁剪面距离，表示从相机出发的光线开始采样的起始位置。这个值应该根据场景的实际情况进行调整，通常设置为2.0或更大，以避免采样到相机内部的区域。
		far=6.0,  # 远裁剪面距离，表示从相机出发的光线结束采样的位置。这个值应该根据场景的实际情况进行调整，通常设置为6.0或更大，以确保采样到场景中的所有物体。
		multires=10,  # 位置编码的频率数量，通常是10或15。
		multires_views=4,  # 视角编码的频率数量，通常是4或6。
		test_render_batch=4096,  # 渲染测试图时每次处理的光线数量，避免显存峰值过高。
		lrate=5e-4,  # 初始学习率，通常是5e-4或1e-3。这个值越大，模型的收敛速度越快，但可能会导致训练不稳定；这个值越小，模型的收敛速度越慢，但训练更稳定。可以根据训练情况进行调整。
		lrate_decay=250000,  # 学习率衰减的迭代次数，通常是250000或500000。这个值越大，学习率衰减得越慢，模型的拟合能力越强，但训练时间也会更长；这个值越小，学习率衰减得越快，模型的拟合能力越弱，但训练时间也会更短。可以根据训练情况进行调整。
		i_print=100,  # 每隔多少迭代打印一次训练日志，通常是100或500。这个值越小，训练日志越详细，但会增加训练时间；这个值越大，训练日志越简洁，但可能会错过一些重要的训练信息。可以根据需要进行调整。
		i_weights=5000,  # 每隔多少迭代保存一次模型权重，通常是5000或10000。这个值越小，模型权重保存得越频繁，可以更好地记录训练过程，但会占用更多的存储空间；这个值越大，模型权重保存得越不频繁，可以节省存储空间，但可能会错过一些重要的训练阶段。可以根据需要进行调整。
	)
	train(args)
