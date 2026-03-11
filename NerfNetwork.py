import torch
import torch.nn as nn
import torch.nn.functional as F

#! NeRF网络的定义，输入位置编码和视角编码，输出RGB颜色和密度值。
class NeRF(nn.Module):
    def __init__(self):
        super(NeRF, self).__init__()#先把父类该做的初始化做完，避免父类成员没被正确初始化
        #? Nerf的输入分为两块，位置编码和视角编码，位置编码的维度为63，视角编码的维度为27。
        self.input_ch = 63
        self.input_ch_views = 27
        #? 定义输入层：输入位置编码，经过5层全连接网络，每层256个神经元，激活函数为ReLU。随后合并视角编码，接着继续经过三个全连接层
        # 第一个全连接层
        self.fc1 = nn.Linear(self.input_ch, 256)
        # 第二个全连接层
        self.fc2 = nn.Linear(256, 256)
        # 第三个全连接层
        self.fc3 = nn.Linear(256, 256)
        # 第四个全连接层
        self.fc4 = nn.Linear(256, 256)
        # 第五个全连接层
        self.fc5 = nn.Linear(256, 256)
        # 第六个全连接层，第五层输出256合并位置编码后输入到第六层
        # 这么做的目的应该是为了让网络更好地利用位置信息，增强对空间结构的理解能力。
        self.fc6 = nn.Linear(256 + self.input_ch, 256)
        # 第七个全连接层
        self.fc7 = nn.Linear(256, 256)
        # 第八个全连接层
        self.fc8 = nn.Linear(256, 256)
        # 第八个全连接层的输出分为两部分：
        #? 一部分直接一个全连接层之后输出密度值
        self.fc_density = nn.Linear(256, 1)  # 输出密度值
        #? 另一部分经过一个全连接层之后和视角编码合并，经过两个全连接层之后输出RGB颜色值
        self.fc_9 = nn.Linear(256, 256)  # 提取特征
        self.fc_10 = nn.Linear(256 + self.input_ch_views, 128)  # 合并视角编码
        self.fc_rgb = nn.Linear(128, 3)  # 输出RGB颜色值
        

    def forward(self, x):
        # 将输入分为位置编码和视角编码两部分
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        # 位置编码部分经过五层全连接网络
        h = F.relu(self.fc1(input_pts))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        h = F.relu(self.fc5(h))
        # 合并视角编码
        h = torch.cat([h, input_pts], dim=-1)
        # 经过第六层全连接网络
        h = F.relu(self.fc6(h))
        # 经过第七层全连接网络
        h = F.relu(self.fc7(h))
        # 经过第八层全连接网络
        h = F.relu(self.fc8(h))
        # 输出密度值
        density = self.fc_density(h)
        # 输出RGB颜色值
        h = F.relu(self.fc_9(h))
        h = torch.cat([h, input_views], dim=-1)
        h = F.relu(self.fc_10(h))
        rgb = self.fc_rgb(h)
        #? 合并密度值和RGB颜色值
        output = torch.cat([rgb, density], dim=-1)

        return output


# 主函数用于测试NeRF网络
if __name__ == "__main__":
    model = NeRF()
    # print(model)
    x = torch.randn(1, 63 + 27)  # 随机生成一个输入样本，假设位置编码后维度为63，视角编码后维度为27
    output = model(x)
    print(f"Output shape: {output.shape}")  # 输出的形状应为 (1, 4)
    model.eval()
    # 使用onnx导出模型。新导出器默认以较新opset生成图，避免再降级转换导致Relu适配器错误。
    torch.onnx.export(
        model,
        x,
        "nerf_model.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
    )