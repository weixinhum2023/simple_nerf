# Author: weixihum
# Date: 2026-03-10
# 读取blender数据集的图像数据，并将其转化为起点+方向+RGB的形式，方便后续训练使用。
# 文件采用了Open3D进行可视化，以便我们更好地理解数据集中的相机位姿和光线分布。

import torch
import numpy as np
import json
import os
import imageio
import open3d as o3d


#! 生成相机所在的位置，参数为球坐标的三个参数。
def pose_spherical(theta, phi, radius):
    # 平移矩阵
    def trans_t(t):
        return torch.Tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]
        ).float()

    # 旋转矩阵,绕x轴旋转phi度
    def rot_phi(phi):
        return torch.Tensor(
            [
                [1, 0, 0, 0],
                [0, np.cos(phi), -np.sin(phi), 0],
                [0, np.sin(phi), np.cos(phi), 0],
                [0, 0, 0, 1],
            ]
        ).float()

    # 旋转矩阵,绕y轴旋转theta度
    def rot_theta(th):
        return torch.Tensor(
            [
                [np.cos(th), 0, -np.sin(th), 0],
                [0, 1, 0, 0],
                [np.sin(th), 0, np.cos(th), 0],
                [0, 0, 0, 1],
            ]
        ).float()

    c2w = trans_t(radius)  # 首先将相机沿着z轴平移到距离原点radius的位置
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w  # 然后绕x轴旋转phi度
    c2w = (
        rot_theta(theta / 180.0 * np.pi) @ c2w
    )  # 最后绕y轴旋转theta度，得到相机的位姿矩阵
    c2w = (
        torch.Tensor(
            np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        )
        @ c2w
    )  # 这里是为了将相机坐标系从OpenGL的右手坐标系转换为我们常用的左手坐标系，具体来说就是交换y轴和z轴，并且将x轴取反。
    return c2w


#! 数据集的加载函数，输入参数为数据集所在的目录、以及测试集跳帧的数量。
#! 输出的imgs是所有图片的数组，poses是拍摄图片的相机外参
#! [H, W, focal]是图片的高、宽和焦距，i_split是一个列表，包含了训练集、验证集和测试集的索引范围。
def load_blender_data(basedir, testskip=1):
    splits = ["train", "val", "test"]
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, "transforms_{}.json".format(s)), "r") as fp:
            metas[s] = json.load(fp)  # 读取json文件

    all_imgs = []
    all_poses = []
    counts = [0]
    # 读取'train'、'val'、'test'三个数据集
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s == "train" or testskip == 0:
            skip = 1  # 训练集不跳帧
        else:
            skip = (
                testskip  # 验证集和测试集可能跳帧，需要查看设置，lego这个模型也是不跳的
            )

        for frame in meta[
            "frames"
        ][
            ::skip
        ]:  # Python 的切片 [start:stop:step] 形式。::skip 等价于从头到尾、步长为 skip 的切片
            fname = os.path.join(basedir, frame["file_path"] + ".png")
            imgs.append(
                imageio.v3.imread(fname)
            )  # 读取图片，得到一个H×W×3的数组，表示图片的像素值
            poses.append(np.array(frame["transform_matrix"]))  # 读取外参
        imgs = (np.array(imgs) / 255.0).astype(
            np.float32
        )  # 这里除以255是为了归一化到0-1之间
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])  # 记录数据集的图片数量
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [
        np.arange(counts[i], counts[i + 1]) for i in range(3)
    ]  # 后续可以用 i_split[0] 取出训练集图片，i_split[1] 取验证集，i_split[2] 取测试集，非常方便地分割数据。

    imgs = np.concatenate(
        all_imgs, 0
    )  # 将训练集、验证集、测试集的图片和外参分别连接起来
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]  # shape[:2] 取该图片的前两个维度（高和宽）
    camera_angle_x = float(meta["camera_angle_x"])  # 获得相机的水平视角
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)  # 根据视角和宽度计算焦距

    # ? 将原来的黑色背景改为白色，进行alpha blending。
    # 对于RGBA格式的图片，imgs[...,-1:]会得到一个形状为(H, W, 1)的数组，表示每个像素的alpha值。
    # imgs[...,:3]会得到一个形状为(H, W, 3)的数组，表示每个像素的RGB值。
    # 通过乘以alpha值和加上背景颜色，可以得到最终的RGB值。
    imgs = imgs[..., :3] * imgs[..., -1:] + (1.0 - imgs[..., -1:])

    # # 显示第一张图片
    # import matplotlib.pyplot as plt
    # plt.imshow(imgs[0])
    # plt.axis('off')
    # plt.show()

    return imgs, poses, int(H), int(W), focal, i_split


#! 生成相机渲染视角，用于生成训练之后的视频，参数num_poses表示生成的视角数量，radius表示相机距离原点的距离。
def generate_render_poses(num_poses=40, radius=4.0):
    # np.linspace(-180,180,40+1)[:-1]表示在区间 [−180,180] 上生成 41 个等间隔的浮点数。[:-1]作用是去掉最后一个元素，也就是180。
    render_poses = torch.stack(
        [
            pose_spherical(angle, -30.0, radius)
            for angle in np.linspace(-180, 180, num_poses + 1)[:-1]
        ],
        0,
    )
    return render_poses


#! 可视化数据加载结果的函数
def visualize_data(poses, K, i_split):
    # 遍历所有的相机位姿
    camera_frames = []
    for i in range(poses.shape[0]):
        w2c_opengl = np.linalg.inv(poses[i])
        flip_yz = np.eye(4)
        flip_yz[1, 1] = -1
        flip_yz[2, 2] = -1
        extrinsic = flip_yz @ w2c_opengl
        intrinsic = K
        cameraLines = o3d.geometry.LineSet.create_camera_visualization(
            view_width_px=int(K[0, 2] * 2),
            view_height_px=int(K[1, 2] * 2),
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            scale=0.2,
        )

        if i in i_split[0]:  # 训练集为蓝色
            cameraLines.paint_uniform_color((0, 0, 1))  # Blue
        elif i in i_split[1]:  # 验证集为黄色
            cameraLines.paint_uniform_color((1, 1, 0))  # Yellow
        else:  # 测试集为红色
            cameraLines.paint_uniform_color((1, 0, 0))  # Red
        camera_frames.append(cameraLines)

    vizualizer = o3d.visualization.Visualizer()
    vizualizer.create_window(window_name="Camera Pose", width=800, height=600)
    for camera in camera_frames:
        vizualizer.add_geometry(camera)
    vizualizer.run()
    vizualizer.destroy_window()


#! 显示用到的库的版本
def print_library_versions():
    print("PyTorch version:", torch.__version__)
    print("Open3D version:", o3d.__version__)
    print("NumPy version:", np.__version__)
    print("ImageIO version:", imageio.__version__)


# 依据图像的高、宽、相机内参矩阵和相机外参矩阵，生成每个像素对应的光线起点和方向。
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing="ij"
    )  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    # 把相机中心当原点，成像平面放在 (z=-f)。
    # 像素 ((u,v)) 先减主点 ((c_x,c_y))，得到相对光轴的偏移。
    # 再除以焦距作用是z直接等于-1，得到相机坐标系下的光线方向。y前面加了负号是因为图像坐标系的y轴是向下的，而相机坐标系的y轴是向上的，所以需要取反。
    dirs = torch.stack(
        [(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1
    )  # dim=-1 表示在最后新增一维进行堆叠，这样得到的dirs的形状是(H, W, 3)，每个像素对应一个三维向量，表示相机坐标系下的光线方向。
    # 这里的dirs是相机坐标系下的光线方向，c2w[:3,:3]是相机外参矩阵中的旋转部分，将相机坐标系下的光线方向旋转到世界坐标系下。
    rays_d = torch.sum(
        dirs[..., np.newaxis, :] * c2w[:3, :3], -1
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    # 这里的c2w[:3,-1]是相机外参矩阵中的平移部分，表示相机在世界坐标系中的位置。expand(rays_d.shape)是为了将这个位置扩展到和光线方向数组的形状一致，以便后续计算。
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


#! 将图像处理为光线起点+终点+RGB的形式
def process_image_to_rays(img, pose, K):
    # 输入一张图像，得到该图像中每个像素对应的光线起点、光线方向和RGB值。输出的rays_o、rays_d和rgbs的形状分别是 (H*W, 3)，(H*W, 3)，(H*W, 3)，其中 H 和 W 是图像的高和宽。

    H, W = img.shape[:2]
    rays_o, rays_d = get_rays(
        H, W, K, pose
    )  # 得到每个像素对应的光线起点和方向，形状是 (H, W, 3)
    rays_o = rays_o.reshape(-1, 3)  # 将光线起点数组展平为 (H*W, 3)
    rays_d = rays_d.reshape(-1, 3)  # 将光线方向数组展平为 (H*W, 3)
    rgbs = img.reshape(-1, 3)  # 将图像的RGB值展平为 (H*W, 3)
    return rays_o, rays_d, rgbs


#! 可视化光线的函数，输入参数为光线起点、光线方向和RGB值。这个函数会在Open3D中创建一个点云对象，每个点的位置由光线起点加上光线方向乘以一个缩放因子得到，颜色由RGB值决定。
def visualize_rays(rays_o, rays_d, rgbs):
    # 使用Open3D可视化光线起点和方向，沿着光线方向绘制线段，长度为1，颜色为RGB值。
    points = rays_o.cpu().numpy()  # 光线起点
    directions = rays_d.cpu().numpy()  # 光线方向
    colors = rgbs.cpu().numpy()  # RGB颜色

    ray_len = 1.0
    end_points = points + directions * ray_len
    all_points = np.concatenate([points, end_points], axis=0)
    num_rays = points.shape[0]
    lines = np.stack([np.arange(num_rays), np.arange(num_rays) + num_rays], axis=1)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(all_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)  # 每条线段连接起点和终点
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([line_set])


# 主函数
if __name__ == "__main__":
    #!############## 显示用到的库的版本 ##############
    # print_library_versions()
    #!################## 数据加载 ###################
    datadir = "./lego"
    images, poses, H, W, focal, i_split = load_blender_data(datadir, testskip=8)
    # 构建相机内参矩阵，0.5*W和0.5*H是图像的中心点坐标，focal是焦距。
    K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])
    # visualize_data(poses, K, i_split)# 相机视锥可视化
    #!###### 将图像处理为光线起点+终点+RGB的形式 #######
    # 先将poses和K转换为torch.Tensor格式
    poses = torch.from_numpy(poses).float()
    K = torch.from_numpy(K).float()
    # 处理第一张图像，得到光线起点、方向和RGB值
    rays_o, rays_d, rgbs = process_image_to_rays(
        torch.from_numpy(images[0]).float(), poses[0], K
    )
    visualize_rays(rays_o, rays_d, rgbs)  # 光线可视化
