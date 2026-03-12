import torch

def positional_encoding(inputs, multires, use_encoding=True):
    """直接对输入做位置编码。

    参数:
        inputs: 原始输入张量，形状为 [..., C]。
        multires: 频率数量，用于生成 [1, 2, 4, ..., 2^(multires-1)]。
        use_encoding: 是否进行位置编码。

    返回:
        编码后的张量；当 use_encoding=False 时直接返回 inputs。
    """
    if not use_encoding:
        return inputs

    frequencies = 2.0 ** torch.arange(
        multires, dtype=inputs.dtype, device=inputs.device
    )

    encoded_parts = [inputs]
    for freq in frequencies:
        encoded_parts.append(torch.sin(inputs * freq))
        encoded_parts.append(torch.cos(inputs * freq))
    return torch.cat(encoded_parts, dim=-1)


# 主函数用于测试位置编码器
if __name__ == "__main__":
    # 生成一个测试输入点 (1, 3)
    x = torch.tensor([[0.1, 0.2, 0.3]])
    embedded_x = positional_encoding(x, multires=10, use_encoding=True)
    print(f"Embedded output shape: {embedded_x.shape}")  # 输出编码结果的形状
    print(f"Embedded output: {embedded_x}")  # 输出编码结果
