from math import sqrt

import einops
from einops.layers.torch import Reduce
import torch
import torch.nn as nn


class LinearEmbedding(nn.Sequential):
    """
    這個類別是用來將輸入的訊號轉換成 Embedding
    """
    def __init__(self, input_channels, output_channels) -> None:
        # Input Embedding Layer, output channel 為 embed_size
        super().__init__(*[
            nn.Linear(input_channels, output_channels),
            nn.LayerNorm(output_channels),
            nn.GELU()
        ])
        # position embedding
        self.cls_token = nn.Parameter(torch.randn(1, output_channels))

    def forward(self, x):
        # 直接使用上面創建的 Linear -> LayerNorm -> GELU()
        embedded = super().forward(x)
        # repeat(self, pattern, **axes_lengths) 產生符合 batch size 的 token
        return torch.cat([einops.repeat(self.cls_token, "n e -> b n e", b=x.shape[0]), embedded], dim=1)


class MLP(nn.Sequential):
    def __init__(self, input_channels, expansion=4):
        super().__init__(*[
            nn.Linear(input_channels, input_channels * expansion),
            nn.GELU(),
            nn.Linear(input_channels * expansion, input_channels)
        ])


class ResidualAdd(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_size, num_heads, attention_store=None):
        super().__init__()
        self.queries_projection = nn.Linear(embed_size, embed_size)
        self.values_projection = nn.Linear(embed_size, embed_size)
        self.keys_projection = nn.Linear(embed_size, embed_size)
        self.final_projection = nn.Linear(embed_size, embed_size)
        self.embed_size = embed_size
        self.num_heads = num_heads

    def forward(self, x):
        """
        :param x: Size=[batch, seq_len, embed_size]
        :return:
        """
        assert len(x.shape) == 3
        keys = self.keys_projection(x)
        # print(f"keys: {keys.shape}")
        values = self.values_projection(x)
        queries = self.queries_projection(x)
        # 將 embed_size 拆成 num_heads 個部分
        keys = einops.rearrange(keys, "b n (h e) -> b n h e", h=self.num_heads)
        queries = einops.rearrange(queries, "b n (h e) -> b n h e", h=self.num_heads)
        values = einops.rearrange(values, "b n (h e) -> b n h e", h=self.num_heads)
        # 針對每個 head 做 attention 運算
        energy_term = torch.einsum("bqhe, bkhe -> bqhk", queries, keys)
        # To reduce variance, therefore divide by the square root of the dimension
        divider = sqrt(self.embed_size)
        mh_out = torch.softmax(energy_term, -1)
        out = torch.einsum('bihv, bvhd -> bihd ', mh_out / divider, values)
        out = einops.rearrange(out, "b n h e -> b n (h e)")
        return self.final_projection(out)


class TransformerEncoderLayer(torch.nn.Sequential):
    def __init__(self, embed_size=768, expansion=4, num_heads=4, dropout=0.1):
        """
        :param embed_size: 原本是 768
        :param expansion:  原本是 4
        :param num_heads:  原本是 6
        :param dropout:
        """
        super(TransformerEncoderLayer, self).__init__(
            *[
                ResidualAdd(nn.Sequential(*[
                    nn.LayerNorm(embed_size),
                    MultiHeadAttention(embed_size, num_heads),
                    nn.Dropout(dropout)
                ])),
                ResidualAdd(nn.Sequential(*[
                    nn.LayerNorm(embed_size),
                    MLP(embed_size, expansion),
                    nn.Dropout(dropout)
                ]))
            ]
        )


class Classifier(nn.Sequential):
    def __init__(self, embed_size, num_classes):
        super().__init__(*[
            Reduce("b n e -> b e", reduction="mean"),
            nn.Linear(embed_size, embed_size),
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, num_classes)
        ])


class ECGformer(nn.Module):

    def __init__(self, num_layers, signal_length, num_classes, input_channels, embed_size, num_heads, expansion) -> None:
        """
        :param num_layers: 重複 N 次
        :param signal_length: 輸入訊號的長度
        :param num_classes: 輸出的分類
        :param input_channels: 輸入訊號的通道數
        :param embed_size: 這個參數控制了 Transformer 的維度大小，通常情況下，這個參數的值越大，模型的性能也會越好，
        :param num_heads: 這個參數控制了 MultiHeadAttention 的頭數，頭數越多，模型可以學習到更多的特徵表示.
        :param expansion: 在 MLP 的結構中，首先有一個線性層將輸入的維度擴展到 input_channels * expansion，
                          然後經過一個 GELU 激活函數，最後再通過一個線性層將維度壓縮回 input_channels。
                          這種結構有時被稱為 "bottleneck" 結構，因為它先擴展維度，然後再壓縮維度。
                          這種結構的好處是，當 expansion 大於 1 時，模型可以在隱藏層中學習更多的特徵表示，
                          這可能有助於提高模型的性能。然而，這也會增加模型的參數數量和計算量，
                          因此需要根據具體的應用場景來選擇合適的 expansion 值
        """
        super().__init__()
        self.encoder = nn.ModuleList([TransformerEncoderLayer(
            embed_size=embed_size, num_heads=num_heads, expansion=expansion) for _ in range(num_layers)])
        self.classifier = Classifier(embed_size, num_classes)
        self.positional_encoding = nn.Parameter(torch.randn(signal_length + 1, embed_size))
        self.embedding = LinearEmbedding(input_channels, embed_size)

    def forward(self, x):
        embedded = self.embedding(x)

        for layer in self.encoder:
            embedded = layer(embedded + self.positional_encoding)


        return self.classifier(embedded)


if __name__ == "__main__":
    from torchinfo import summary
    # print(LinearEmbedding(1, 192)(torch.rand(2, 128, 1)).shape)
    # print(MLP(3)(torch.rand(2, 128, 3)).shape)
    # print(TransformerEncoderLayer(192, 8)(torch.rand(2, 128, 192)).shape)
    print(ECGformer(6, 187, 2, 1, 192, 8, 4)(torch.rand(2, 187, 1)).shape)
    # print(MultiHeadAttention(192, 8)(torch.rand(1, 186, 192)).shape)





