import torch

def patchify_and_mask(images, patch_size=14, mask_ratio=0.75):
    """
    对一个批次的图片进行分块和随机掩码。

    Args:
        images (torch.Tensor): 输入的图片张量，形状为 (B, C, H, W)，例如 (B, 3, 224, 224)。
        patch_size (int): 每个块的大小，例如 16。
        mask_ratio (float): 掩码的比例，例如 0.75 表示遮盖掉 75% 的块。

    Returns:
        tuple: 包含以下元素的元组:
            - visible_patches (torch.Tensor): 经过掩码后仍然可见的块。
            - mask (torch.Tensor): 二进制掩码，1表示被遮盖，0表示可见。
            - ids_restore (torch.Tensor): 用于在解码器中恢复原始顺序的索引。
    """
    B, C, H, W = images.shape
    assert H % patch_size == 0 and W % patch_size == 0, '图片尺寸必须能被patch_size整除'

    # 1. 图片分块 (Patchify)
    # (B, C, H, W) -> (B, N, L)
    # N 是块的数量, L 是每个块展平后的向量长度
    num_patches = (H // patch_size) * (W // patch_size)
    patch_dim = C * patch_size * patch_size

    # 将图片重塑并切分成块
    # (B, C, H, W) -> (B, C, num_patches_h, patch_size, num_patches_w, patch_size)
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
    # (B, num_patches_h, num_patches_w, C, patch_size, patch_size) -> (B, N, L)
    patches = patches.view(B, num_patches, patch_dim)

    # 2. 生成随机掩码 (Generate Mask)
    len_keep = int(num_patches * (1 - mask_ratio)) # 需要保留的块的数量

    # 为每个样本生成一个随机打乱的索引序列
    noise = torch.rand(B, num_patches, device=images.device)  # (B, N)
    ids_shuffle = torch.argsort(noise, dim=1)  # 随机打乱索引
    ids_restore = torch.argsort(ids_shuffle, dim=1) # 用于恢复原始顺序的索引

    # 决定保留哪些块
    ids_keep = ids_shuffle[:, :len_keep] # (B, len_keep)

    # 3. 应用掩码 (Apply Mask)
    # 根据 ids_keep 来选择可见的块
    # torch.gather 需要的索引维度与输入相同，所以我们需要扩展 ids_keep
    ids_keep_expanded = ids_keep.unsqueeze(-1).expand(-1, -1, patch_dim)
    visible_patches = torch.gather(patches, dim=1, index=ids_keep_expanded)

    # 4. 生成二进制掩码 (用于解码器)
    mask = torch.ones(B, num_patches, device=images.device) # 默认所有块都被遮盖(1)
    mask[:, :len_keep] = 0 # 将保留的块标记为可见(0)
    # 按照恢复顺序重新排列掩码，使其与原始图片块顺序对应
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return visible_patches, mask, ids_restore

# --- 示例使用 ---
if __name__ == '__main__':
    # 创建一个假的图片批次 (2张图片, 3通道, 224x224)
    dummy_images = torch.randn(2, 3, 224, 224)

    # 设置参数
    p_size = 16
    m_ratio = 0.75

    # 执行分块和掩码
    visible_patches, mask, ids_restore = patchify_and_mask(dummy_images, patch_size=p_size, mask_ratio=m_ratio)

    # 打印结果的形状
    num_patches = (224 // p_size) ** 2
    len_keep = int(num_patches * (1 - m_ratio))

    print(f"原始图片形状: {dummy_images.shape}")
    print("-" * 30)
    print(f"Patch大小: {p_size}x{p_size}")
    print(f"总块数 (N): {num_patches}")
    print(f"掩码率: {m_ratio * 100}%")
    print(f"保留块数: {len_keep}")
    print("-" * 30)
    print(f"可见块的形状 (B, len_keep, L): {visible_patches.shape}")
    print(f"二进制掩码的形状 (B, N): {mask.shape}")
    print(f"恢复索引的形状 (B, N): {ids_restore.shape}")

    # 验证二进制掩码
    # 被遮盖的块的数量应该等于 N * mask_ratio
    num_masked = torch.sum(mask) / dummy_images.shape[0] # 计算单张图片的平均掩码数
    print(f"\n根据二进制掩码计算出的被遮盖块数: {int(num_masked)}")
    print(f"理论上应被遮盖的块数: {num_patches - len_keep}")