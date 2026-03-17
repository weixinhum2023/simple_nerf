import os
import time
from collections import deque
from types import SimpleNamespace

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F

from Embedder import positional_encoding
from NerfNetwork import NeRF
from ReadData import load_blender_data
from VolumeRendering import raw2outputs


def format_seconds(seconds):
	seconds = max(0, int(seconds))
	hours = seconds // 3600
	minutes = (seconds % 3600) // 60
	secs = seconds % 60
	return f"{hours:02d}:{minutes:02d}:{secs:02d}"

#! 对采样点编码后一次性送入网络，得到预测的RGB颜色值和密度值。
def run_network(inputs, viewdirs, model, multires, multires_views):
	input_flat = inputs.reshape(-1, inputs.shape[-1])# 将输入的采样点位置编码成一个二维张量，第一维是所有采样点的数量，第二维是位置编码后的维度。这个位置编码的过程是为了让网络能够更好地利用空间信息，增强对场景的理解能力。
	embedded_pts = positional_encoding(input_flat, multires=multires, use_encoding=True)

	viewdirs_expanded = viewdirs[:, None, :].expand(inputs.shape)
	viewdirs_flat = viewdirs_expanded.reshape(-1, viewdirs_expanded.shape[-1])
	embedded_views = positional_encoding(viewdirs_flat, multires=multires_views, use_encoding=True)

	network_input = torch.cat([embedded_pts, embedded_views], dim=-1)
	outputs = model(network_input)
	outputs = outputs.reshape(inputs.shape[0], inputs.shape[1], 4)
	return outputs

#! 对每条光线进行体渲染，得到预测的RGB颜色值。
def render_rays(
	rays_o,
	rays_d,
	coarse_model,
	fine_model,
	near,
	far,
	n_samples,
	n_importance,
	multires,
	multires_views,
	white_bkgd,
):
	n_rays = rays_o.shape[0]
	t_vals = torch.linspace(0.0, 1.0, n_samples, device=rays_o.device)
	z_vals_coarse = near * (1.0 - t_vals) + far * t_vals
	z_vals_coarse = z_vals_coarse.expand(n_rays, n_samples)# expand函数的作用是将z_vals_coarse这个一维张量扩展成一个二维张量，第一维的大小是n_rays，第二维的大小是n_samples。每一行都是原来的z_vals，这样每条光线就有了对应的采样点深度值。

	pts_coarse = rays_o[:, None, :] + rays_d[:, None, :] * z_vals_coarse[..., :, None]
	viewdirs = rays_d # rays_d已经是单位向量了，不需要再进行归一化了。

	# 粗网络训练
	raw_coarse = run_network(
		pts_coarse,
		viewdirs,
		coarse_model,
		multires=multires,
		multires_views=multires_views,
	)

	rgb_coarse = raw_coarse[..., :3]
	sigma_coarse = raw_coarse[..., 3]
	coarse_rgb_map, _, weights, _ = raw2outputs(rgb_coarse, sigma_coarse, z_vals_coarse, white_bkgd=white_bkgd)

	# 依据粗网络的权重分布进行重要性采样，得到新的深度点
	z_vals_mid = 0.5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1]) # 计算每个采样点之间的中点位置，作为新的采样点的候选位置。
	z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], n_importance) # 根据权重分布采样新的深度点，得到新的采样点位置。这里的z_vals_mid是每个采样点之间的中点位置，weights[..., 1:-1]是每个采样点的权重（去掉第一个和最后一个采样点，因为它们没有对应的中点），n_importance是要采样的点的数量.
	z_samples = z_samples.detach()  # 与原始NeRF一致：阻断fine损失通过采样路径回传到coarse
	z_vals_fine = torch.sort(torch.cat([z_vals_coarse, z_samples], -1), -1)[0] # 将原来的采样点和新的采样点合并，并按照深度值进行排序，得到最终的采样点位置.
	pts_fine = rays_o[:, None, :] + rays_d[:, None, :] * z_vals_fine[..., :, None] # 根据新的采样点位置计算对应的三维空间坐标。

	# 精细网络训练
	raw_fine = run_network(
		pts_fine,
		viewdirs,
		fine_model,
		multires=multires,
		multires_views=multires_views,
	)
	rgb_fine = raw_fine[..., :3]
	sigma_fine = raw_fine[..., 3]
	fine_rgb_map, _, _, _ = raw2outputs(rgb_fine, sigma_fine, z_vals_fine, white_bkgd=white_bkgd)

	return coarse_rgb_map, fine_rgb_map

#! 按小批量光线渲染整张测试图，避免显存峰值过高。
def render_test_image(coarse_model, fine_model, rays_o, rays_d, H, W, args):
	outputs_coarse = []
	outputs_fine = []
	with torch.no_grad():
		for start in range(0, rays_o.shape[0], args.test_render_batch):
			end = start + args.test_render_batch
			pred_rgb_coarse, pred_rgb_fine = render_rays(
				rays_o[start:end],
				rays_d[start:end],
				coarse_model,
				fine_model,
				near=args.near,
				far=args.far,
				n_samples=args.n_samples,
				n_importance=args.N_importance,
				multires=args.multires,
				multires_views=args.multires_views,
				white_bkgd=True,
			)
			outputs_coarse.append(pred_rgb_coarse)
			outputs_fine.append(pred_rgb_fine)
	img_coarse = torch.cat(outputs_coarse, dim=0).reshape(H, W, 3)
	img_fine = torch.cat(outputs_fine, dim=0).reshape(H, W, 3)
	return torch.clamp(img_coarse, 0.0, 1.0), torch.clamp(img_fine, 0.0, 1.0)


#! 从权重分布中采样新的深度点
def sample_pdf(bins, weights, N_importance):
	# Get pdf
	weights = weights + 1e-5 # prevent nans
	pdf = weights / torch.sum(weights, -1, keepdim=True) # 权重归一化，得到概率密度函数
	cdf = torch.cumsum(pdf, -1) # 累计求和，得到累积分布函数
	cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1) # (batch, len(bins)) # 在前面添加0，表示cdf从0开始
	u = torch.rand(list(cdf.shape[:-1]) + [N_importance], device=cdf.device) # 在[0,1]区间内均匀采样N_importance个点
	# Invert CDF（用CDF反采样，定位u落在哪个区间）
	u = u.contiguous()  # 确保内存连续，便于后续索引运算
	inds = torch.searchsorted(cdf, u, right=True) # 找到每个 u 落在哪个 CDF 区间
	below = torch.max(torch.zeros_like(inds-1), inds-1) # 计算下界索引，避免越界
	above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds) # 计算上界索引，避免越界
	inds_g = torch.stack([below, above], -1) # 得到点的上下界索引对

	matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]] # 让CDF/Bins扩展到可按inds_g索引的形状
	cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g) # 取上下界CDF值
	bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g) # 取上下界深度值（bin）

	denom = (cdf_g[...,1]-cdf_g[...,0]) # 计算区间占比，得到概率密度
	denom = torch.where(denom<1e-5, torch.ones_like(denom), denom) # 防止除零
	t = (u-cdf_g[...,0])/denom # 计算u在区间内的归一化位置
	samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0]) # 采样
	return samples


#! 训练函数，包含训练循环、日志记录、模型保存和测试渲染等功能。
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
	# 创建粗网络并移动到设备上
	coarse_model = NeRF().to(device)
	optimizer_coarse = torch.optim.Adam(coarse_model.parameters(), lr=args.lrate)
	fine_model = NeRF().to(device)
	optimizer_fine = torch.optim.Adam(fine_model.parameters(), lr=args.lrate)
    # 创建日志文件夹和日志文件路径
	os.makedirs(args.basedir, exist_ok=True)
	preview_dir = os.path.join(args.basedir, "test_renders")
	os.makedirs(preview_dir, exist_ok=True)
	log_path = os.path.join(args.basedir, "train_log.txt")

	i_test = i_split[2]
	# 如果测试集不为空，则使用测试集的第一张图像进行测试渲染；如果测试集为空，则使用训练集的第一张图像进行测试渲染。
	test_img_i = int(i_test[0]) if len(i_test) > 0 else int(i_train[0])
	test_rays_o = rays_o_all[test_img_i]
	test_rays_d = rays_d_all[test_img_i]
    # 训练循环，迭代次数为args.n_iters，每次迭代随机选取一张图像进行训练
	recent_step_times = deque(maxlen=100)
	t0 = time.time()
	for step in range(1, args.n_iters + 1):
		step_start = time.time()
		# 随机选取一张图像，并从中随机选取args.n_rand条光线进行训练
		img_i = np.random.choice(i_train)
		
		# 前500轮只从图像中心区域选择光线
		if step <= args.precrop_iters:
			center_h_ratio = 0.5  # 中心50%的高度
			center_w_ratio = 0.5  # 中心50%的宽度
			
			h_start = int(H * (1 - center_h_ratio) / 2)
			h_end = int(H * (1 + center_h_ratio) / 2)
			w_start = int(W * (1 - center_w_ratio) / 2)
			w_end = int(W * (1 + center_w_ratio) / 2)
			
			h_inds = torch.arange(h_start, h_end, device=device)
			w_inds = torch.arange(w_start, w_end, device=device)
			hh, ww = torch.meshgrid(h_inds, w_inds, indexing='ij')
			
			center_inds = (hh * W + ww).reshape(-1)
			rand_idx = torch.randint(0, len(center_inds), (args.n_rand,), device=device)
			select_inds = center_inds[rand_idx]
		else:
			# 500轮后从全图选择
			select_inds = torch.randint(0, n_rays_per_image, (args.n_rand,), device=device)
		
		rays_o = rays_o_all[img_i, select_inds]
		rays_d = rays_d_all[img_i, select_inds]
		target_rgb = rgbs_all[img_i, select_inds]
        # 对选取的光线进行渲染，得到预测的RGB颜色值
		pred_rgb_coarse, pred_rgb_fine = render_rays(
			rays_o,
			rays_d,
			coarse_model,
			fine_model,
			near=args.near,
			far=args.far,
			n_samples=args.n_samples,
			n_importance=args.N_importance,
			multires=args.multires,
			multires_views=args.multires_views,
			white_bkgd=True,
		)
        # 计算预测的RGB颜色值与目标RGB颜色值之间的均方误差损失，并计算PSNR值
		loss_coarse = F.mse_loss(pred_rgb_coarse, target_rgb)
		psnr_coarse = -10.0 * torch.log10(loss_coarse.detach())

		loss_fine = F.mse_loss(pred_rgb_fine, target_rgb)
		psnr_fine = -10.0 * torch.log10(loss_fine.detach())
        # 反向传播和优化
		optimizer_coarse.zero_grad()
		optimizer_fine.zero_grad()
		total_loss = loss_coarse + loss_fine
		total_loss.backward()
		optimizer_coarse.step()
		optimizer_fine.step()
		recent_step_times.append(time.time() - step_start)
        # 学习率衰减，按照指数衰减的方式调整学习率，衰减的速度由args.lrate_decay控制
		new_lrate = args.lrate * (0.1 ** (step / args.lrate_decay))
		for param_group in optimizer_coarse.param_groups:
			param_group["lr"] = new_lrate
		for param_group in optimizer_fine.param_groups:
			param_group["lr"] = new_lrate
        # 每隔args.i_print迭代打印一次训练日志，日志内容包括当前迭代次数、损失值、PSNR值、学习率和每次迭代的时间
		if step % args.i_print == 0 or step == 1:
			dt = time.time() - t0
			steps_left = args.n_iters - step
			avg_step_time = sum(recent_step_times) / len(recent_step_times)
			eta_seconds = avg_step_time * steps_left
			msg = (
				f"[Step {step:06d}] Loss_coarse={loss_coarse.item():.6f} PSNR_coarse={psnr_coarse.item():.2f} "
				f"Loss_fine={loss_fine.item():.6f} PSNR_fine={psnr_fine.item():.2f} "
				f"LR={new_lrate:.6e} Time={dt:.2f}s ETA={format_seconds(eta_seconds)}"
			)
			print(msg)
			with open(log_path, "a", encoding="utf-8") as f:
				f.write(msg + "\n")
			t0 = time.time()

		if step in (1, 5000, 10000, 20000, 50000, 100000, 150000, 200000):
			was_coarse_model_training = coarse_model.training
			was_fine_model_training = fine_model.training

			coarse_model.eval()
			fine_model.eval()
			test_img_coarse, test_img_fine = render_test_image(coarse_model, fine_model, test_rays_o, test_rays_d, H, W, args)
			if was_coarse_model_training:
				coarse_model.train()
			if was_fine_model_training:
				fine_model.train()

			out_path = os.path.join(preview_dir, f"test_coarse_step_{step:06d}.png")
			iio.imwrite(out_path, (test_img_coarse.detach().cpu().numpy() * 255.0).astype(np.uint8))
			print(f"Saved test render to {out_path}")

			out_path = os.path.join(preview_dir, f"test_fine_step_{step:06d}.png")
			iio.imwrite(out_path, (test_img_fine.detach().cpu().numpy() * 255.0).astype(np.uint8))
			print(f"Saved test render to {out_path}")
        # # 每隔args.i_weights迭代保存一次模型权重，保存的内容包括当前迭代次数、模型的状态字典和优化器的状态字典，保存的文件名包含当前迭代次数
		# if step % args.i_weights == 0 or step == args.n_iters:
		# 	ckpt_path = os.path.join(args.basedir, f"coarse_{step:06d}.pth")
		# 	torch.save(
		# 		{
		# 			"global_step": step,
		# 			"model_state_dict": coarse_model.state_dict(),
		# 			"optimizer_state_dict": optimizer_coarse.state_dict(),
		# 		},
		# 		ckpt_path,
		# 	)
		# 	print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
	# 训练配置：直接修改这些变量即可
	args = SimpleNamespace(
		datadir="./lego",
		basedir="./logs",
		n_iters=200001,  # 训练迭代次数，一次迭代中会随机选取一张图像进行训练。
		n_rand=1024,  # 每次迭代中随机选取的光线数量。
		precrop_iters=500,  # 预裁剪阶段的迭代次数。这个阶段只训练图像中心区域的光线，可以帮助模型更快地收敛。
		n_samples=64,  # 每条光线上采样的点的数量。
		N_importance=128, # 每条光线的额外精细采样数量，即精细网络比粗网络多采样的点数。
		near=2.0,  # 近裁剪面距离，表示从相机出发的光线开始采样的起始位置。
		far=6.0,  # 远裁剪面距离，表示从相机出发的光线结束采样的位置。
		multires=10,  # 位置编码的频率数量。
		multires_views=4,  # 视角编码的频率数量。
		test_render_batch=4096,  # 渲染测试图时每次处理的光线数量，避免显存峰值过高。
		lrate=5e-4,  # 初始学习率。
		lrate_decay=500000,  # 学习率衰减的迭代次数。
		i_print=500,  # 每隔多少迭代打印一次训练日志。
		i_weights=5000,  # 每隔多少迭代保存一次模型权重。
	)
	train(args)
