# -*- coding: utf-8 -*-
"""
本脚本定义了 ID Encoder (身份编码器)，在 StableAnimator 架构中，它的核心作用是
将一个纯粹的身份特征 (ID Embedding) 与一个包含全局风格、光照、背景信息的
图像特征 (CLIP Embedding) 进行融合，生成一个“精炼后的人脸ID特征”
(Refined Face Embeddings)。

这个过程至关重要，因为它解决了身份保持的一个核心难题：
如何让一个抽象的、与特定图像无关的ID特征，能够适应并融入到任意给定的参考图像
的视觉风格中。

工作流程:
1.  接收一个从人脸识别模型（如 ArcFace）提取的 `id_embeds` (形状: [B, 512])。
2.  接收一个从 CLIP 图像编码器提取的 `clip_embeds` (形状: [B, 257, 1024])。
3.  `FusionFaceId` 类首先将 `id_embeds` 投影并扩展为多个 "ID tokens" (例如4个)。
    这为身份信息提供了更丰富的表达能力。
4.  然后，它利用一个 `FacePerceiver` 模型，该模型的核心是 `PerceiverAttention`。
5.  在 `PerceiverAttention` 中，ID tokens 作为查询（Query），去“感知”和“蒸馏”
    `clip_embeds` 中包含的上下文信息（如风格、光照）。
6.  经过多层处理后，输出的 ID tokens 就吸收了参考图的风格，成为了最终的
    Refined Face Embeddings，供后续的 U-Net ID-Adapter 使用。
"""

import math
import torch
import torch.nn as nn
from diffusers.models.modeling_utils import ModelMixin

def reshape_tensor(x, heads):
    """一个辅助函数，用于为多头注意力机制重塑张量。"""
    bs, length, width = x.shape
    # (bs, length, width) -> (bs, length, heads, head_dim)
    x = x.view(bs, length, heads, -1)
    # (bs, length, heads, head_dim) -> (bs, heads, length, head_dim)
    x = x.transpose(1, 2)
    # (bs, heads, length, head_dim) -> (bs * heads, length, head_dim)
    # 注：原版代码的 reshape 稍有不同，但 transpose 后的形状已适合 attention 计算
    x = x.reshape(bs, heads, length, -1)
    return x

class PerceiverAttention(nn.Module):
    """
    Perceiver 风格的注意力模块。
    它的核心思想是使用一小组“潜在”（latent）查询向量，去高效地从一个
    非常大的输入数据中“感知”和“提取”信息。
    """
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim) # 对输入 x (图像特征)进行层归一化
        self.norm2 = nn.LayerNorm(dim) # 对输入 latents (ID tokens)进行层归一化

        # 线性层：从 latents 生成 Query
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        # 线性层：从 x 和 latents 的拼接中生成 Key 和 Value
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        # 输出线性层
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): 图像特征 (来自CLIP)，作为上下文信息。
                shape: (b, n1, D)，其中 n1 通常是 257。
            latents (torch.Tensor): 潜在特征 (ID tokens)，作为查询。
                shape: (b, n2, D)，其中 n2 通常是 4。
        """
        # 先对输入进行归一化
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape # b: batch_size, l: ID tokens 的数量 (e.g., 4)

        # 1. 从 latents 生成 Q
        q = self.to_q(latents)
        
        # 2. 将图像特征 x 和 ID tokens latents 拼接起来，共同生成 K 和 V
        # 这允许 ID tokens 不仅能关注图像特征，也能关注彼此（自注意力）
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        # 3. 重塑 Q, K, V 以适应多头注意力
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # 4. 计算注意力权重并应用
        # 使用一种特殊的缩放因子
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        # 5. 重塑输出并经过最终的线性层
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)
        return self.to_out(out)
    
def FeedForward(dim, mult=4):
    """一个标准的前馈网络 (Feed-Forward Network)，Transformer 块的另一个组成部分。"""
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )

class FacePerceiver(torch.nn.Module):
    """
    面部感知器模型。它堆叠了多个 PerceiverAttention 和 FeedForward 块，
    形成一个类似 Transformer 的结构，用于逐步提炼 ID tokens。
    """
    def __init__(
        self,
        dim=768,
        depth=4,
        dim_head=64,
        heads=16,
        embedding_dim=1280,
        output_dim=768,
        ff_mult=4,
    ):
        super().__init__()
        
        self.proj_in = torch.nn.Linear(embedding_dim, dim)  # 输入投影层，匹配维度
        self.proj_out = torch.nn.Linear(dim, output_dim)    # 输出投影层
        self.norm_out = torch.nn.LayerNorm(output_dim)      # 输出归一化
        
        # 创建 depth 层 Perceiver 块
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        # 对输出投影层进行零初始化，确保训练初期的稳定性
        nn.init.constant_(self.proj_out.weight, 0)
        if self.proj_out.bias is not None:
            nn.init.constant_(self.proj_out.bias, 0)

    def forward(self, latents, x):
        # latents: ID tokens, x: clip_embeds
        x = self.proj_in(x) # 统一 clip_embeds 的维度
        
        # 循环通过每一层 Perceiver 块
        for attn, ff in self.layers:
            # PerceiverAttention + 残差连接
            latents = attn(x, latents) + latents
            # FeedForward + 残差连接
            latents = ff(latents) + latents
            
        latents = self.proj_out(latents)
        return self.norm_out(latents)


class FusionFaceId(ModelMixin):
    """
    顶层封装类，用于融合面部ID嵌入和CLIP图像嵌入。
    """
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, clip_embeddings_dim=1024, num_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens # 将一个ID向量扩展为多少个 token

        # 投影层：将 512 维的 ID 嵌入投影并扩展为 (num_tokens * cross_attention_dim) 维
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )

        self.norm = torch.nn.LayerNorm(cross_attention_dim)

        # 核心的融合模型
        self.fusion_model = FacePerceiver(
            dim=cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=cross_attention_dim // 64, # 动态计算头数
            embedding_dim=clip_embeddings_dim,
            output_dim=cross_attention_dim,
            ff_mult=4,
        )
    
    def forward(self, id_embeds, clip_embeds, shortcut=False, scale=1.0):
        """
        前向传播。

        Args:
            id_embeds (torch.Tensor): ArcFace 等模型提取的ID特征。 shape: [B, 512]
            clip_embeds (torch.Tensor): CLIP 提取的参考图特征。 shape: [B, 257, 1024]
            shortcut (bool): 是否添加残差连接。
            scale (float): 残差连接的缩放因子。

        Returns:
            torch.Tensor: 融合了风格的精炼版ID特征。 shape: [B, num_tokens, 768]
        """
        # 1. 将原始 ID 嵌入投影成多个 ID tokens
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x) 
        
        # 2. 调用 FacePerceiver 模型进行信息融合
        # x 是 query (ID tokens), clip_embeds 是 context
        out = self.fusion_model(x, clip_embeds)
        
        # 3. 可选的 shortcut 连接
        if shortcut:
            out = x + scale * out
        return out
