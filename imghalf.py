# 使用OpenCV读取lego数据集的图片，并且将图片缩小一半，然后将缩小后的图片保存到新的文件夹中
import os
import cv2
import imageio
import numpy as np


def resize_images(input_dir, output_dir, scale=0.5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            img_path = os.path.join(input_dir, filename)
            img = imageio.imread(img_path)  # 读取图片，保持alpha通道
            img = (np.array(img) / 255.0).astype(np.float32)  # 归一化到0-1之间
            H, W = img.shape[:2]
            new_H, new_W = int(H * scale), int(W * scale)
            resized_img = cv2.resize(
                img, (new_W, new_H), interpolation=cv2.INTER_AREA
            )  # 使用INTER_AREA插值方法缩小图片
            output_path = os.path.join(output_dir, filename)
            imageio.imwrite(
                output_path, (resized_img * 255).astype(np.uint8)
            )  # 保存缩小后的图片，恢复到0-255范围


if __name__ == "__main__":
    input_directory = "lego/test"  # 替换为你的输入文件夹路径
    output_directory = "lego/test_half_res"  # 替换为你的输出文件夹路径
    resize_images(input_directory, output_directory)
    input_directory = "lego/val"  # 替换为你的输入文件夹路径
    output_directory = "lego/val_half_res"  # 替换为你的输出文件夹路径
    resize_images(input_directory, output_directory)
    input_directory = "lego/train"  # 替换为你的输入文件夹路径
    output_directory = "lego/train_half_res"  # 替换为你的输出文件夹路径
    resize_images(input_directory, output_directory)
    # 显示进度
    print("图片缩小完成，已保存到:", output_directory)
