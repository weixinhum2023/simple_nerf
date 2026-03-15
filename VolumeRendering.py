import torch

def compute_dists(z_vals):
    #? 输入采样点到焦点的距离，计算每个采样点之间的距离，最后一个采样点到无穷远的距离用一个大数代替
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # 最后一个采样点到无穷远的距离用一个大数代替
    infinity_pad = torch.full_like(dists[..., :1], 1e10) 
    dists = torch.cat([dists, infinity_pad], dim=-1)
    return dists


def compute_weights(alpha):
    #? 这里输入每个点的透视率值，计算每个点的权重，权重的计算方式是：当前点的alpha值乘以之前所有点的透视率值的累积乘积。具体来说，transmittance是之前所有点的透视率值的累积乘积，alpha * transmittance就是当前点的权重。
    # 需要注意的是，第一个点的权重就是它的alpha值，因为之前没有点了，所以transmittance是1。第二个点的权重是它的alpha值乘以第一个点的透视率值，第三个点的权重是它的alpha值乘以前两个点的透视率值的累积乘积，以此类推。
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1,
    )[..., :-1]
    return alpha * transmittance

#TODO: 体渲染函数，输入光线的位置、方向、颜色和密度，输出最终的颜色
# 体渲染部分，输入的点的rgb和密度sigma，以及对应的深度值z_vals（点距离小孔多远）和光线方向rays_d
# 输出的是渲染结果rgb_map、权重总和weight_sum、每个点的权重weights和深度图depth_map
def raw2outputs(rgb, sigma, z_vals, white_bkgd=False):
    # 计算每个采样点之间的距离，最后一个采样点到无穷远的距离用一个大数代替
    dists = compute_dists(z_vals)
    alpha = 1.0 - torch.exp(-sigma * dists)# 计算每个采样点的alpha值（透射率）
    weights = compute_weights(alpha)# 计算每个点在最终颜色中的权重
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)# 计算最终的RGB颜色值。None的作用就是把每个采样点的标量权重扩成“可乘 RGB 三通道”的形状。
    weight_sum = torch.sum(weights, dim=-1)# 计算权重的总和，注意这里算出来不一定是1。假设所有采样点的alpha值都很小（采样到了空气），那么权重也会很小，权重的总和就会远小于1。
    depth_map = torch.sum(weights * z_vals, dim=-1)# 计算深度图，权重乘以对应的深度值z_vals，然后求和得到每条光线的深度值

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - weight_sum[..., None]) # 如果背景是白色的，那么最终的颜色值需要加上背景颜色乘以未被任何点覆盖的部分。未被任何点覆盖的部分就是1减去权重总和，乘以背景颜色（白色是1.0），得到最终的颜色值。注意，这个是近似方法，因为我们不知道未被任何点覆盖的部分的确切颜色，因此只能假设为背景颜色。

    return rgb_map, weight_sum, weights, depth_map


if __name__ == "__main__":
    # 测试raw2outputs函数
    rgb = torch.tensor([
        [[0.2, -0.1, 0.5], [0.4, 0.0, -0.3], [0.1, 0.3, 0.2]],
        [[-0.2, 0.6, 0.1], [0.5, -0.4, 0.7], [0.0, 0.2, -0.1]],
    ])  # 固定模型输出，2条光线，每条光线3个采样点，每个采样点3个值（RGB）
    sigmas = torch.tensor([[0.8], [1.2]])  # 模拟密度值
    z_vals = torch.tensor([[0.5, 1.0, 1.5], [0.3, 0.8, 1.2]])  # 模拟深度值
    rays_d = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])  # 模拟光线方向

    rgb_map, acc_map, weights, depth_map = raw2outputs(rgb, sigmas, z_vals, white_bkgd=False)

    print(f"RGB Map: {rgb_map}")
    print(f"Accumulated Opacity Map: {acc_map}") # 光线被物体遮挡的程度，值越接近1表示光线被完全遮挡，值越接近0表示光线未被遮挡。
    print(f"Weights: {weights}")
    print(f"Depth Map: {depth_map}")