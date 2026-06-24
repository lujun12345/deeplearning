from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.base.base_model import BaseModel


# EEG专用Transformer（轻量化+时空位置编码）
class EEGTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_classes, seq_len, num_electrodes):
        super().__init__()
        # 时空位置编码
        self.spatial_emb = nn.Embedding(num_electrodes, input_dim)  # 电极空域编码
        self.temporal_emb = nn.Embedding(seq_len, input_dim)  # 时间步时序编码
        # Transformer编码器
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        # 分类头
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x: [batch, input_dim, seq_len] -> [batch, seq_len, input_dim]
        x = x.transpose(1, 2)
        batch_size, seq_len, input_dim = x.shape

        # 生成位置编码
        spatial_ids = torch.arange(self.spatial_emb.num_embeddings, device=x.device)[:input_dim]
        temporal_ids = torch.arange(seq_len, device=x.device)
        spatial_emb = self.spatial_emb(spatial_ids).unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        temporal_emb = self.temporal_emb(temporal_ids).unsqueeze(0).expand(batch_size, -1, input_dim)

        # 叠加位置编码
        x = x + spatial_emb + temporal_emb
        # Transformer编码
        x = self.transformer_encoder(x)
        # 全局平均池化+分类
        x = x.mean(dim=1)
        x = self.fc(x)
        return x.unsqueeze(-1)  # 适配原输出维度


class EEGTransformerModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 保留原配置参数
        self.filter_base = config.network.filter_base
        self.kernel_size = config.network.kernel_size
        self.max_pooling = config.network.max_pooling
        self.num_blocks = config.network.num_blocks
        self.num_channels = config.data_loader.num_channels
        self.num_classes = config.data_loader.num_classes
        # Transformer参数（可配置）
        self.trans_dim = config.network.get('trans_dim', 64)
        self.trans_heads = config.network.get('trans_heads', 4)
        self.trans_layers = config.network.get('trans_layers', 2)

        # 保留原卷积特征提取（生成token）
        if self.num_channels != 1:
            self.mixing_block = nn.Sequential(OrderedDict([
                ('mix_conv', nn.Conv2d(1, self.num_channels, (self.num_channels, 1))),
                ('mix_batchnorm', nn.BatchNorm2d(self.num_channels)),
                ('mix_relu', nn.ReLU())
            ]))

        self.shortcuts = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('shortcut_conv_{}'.format(k), nn.Conv2d(
                    in_channels=self.num_channels if k == 0 else 4 *
                                                                 self.filter_base * (2 ** (k - 1)),
                    out_channels=4 * self.filter_base * (2 ** k),
                    kernel_size=(1, 1)))
            ])) for k in range(self.num_blocks)
        ])

        self.blocks = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv_{}_1".format(k), nn.Conv2d(
                    in_channels=self.num_channels if k == 0 else 4 * self.filter_base *
                                                                 (2 ** (k - 1)),
                    out_channels=self.filter_base * (2 ** k),
                    kernel_size=(1, 1))),
                ("batchnorm_{}_1".format(k), nn.BatchNorm2d(
                    self.filter_base * (2 ** k))),
                ("relu_{}_1".format(k), nn.ReLU()),
                ("conv_{}_2".format(k), nn.Conv2d(
                    in_channels=self.filter_base * (2 ** k),
                    out_channels=self.filter_base * (2 ** k),
                    kernel_size=(1, self.kernel_size),
                    padding=(0, self.kernel_size // 2))),
                ("batchnorm_{}_2".format(k), nn.BatchNorm2d(
                    self.filter_base * (2 ** k))),
                ("relu_{}_2".format(k), nn.ReLU()),
                ("conv_{}_3".format(k), nn.Conv2d(
                    in_channels=self.filter_base * (2 ** k),
                    out_channels=4 * self.filter_base * (2 ** k),
                    kernel_size=(1, 1))),
                ("batchnorm_{}_3".format(k), nn.BatchNorm2d(
                    4 * self.filter_base * (2 ** k)))
            ])) for k in range(self.num_blocks)
        ])
        self.maxpool = nn.MaxPool2d(kernel_size=(1, self.max_pooling))
        self.relu = nn.ReLU()

        # ===================== 核心改造：卷积+Transformer =====================
        conv_out_channels = 4 * self.filter_base * (2 ** (self.num_blocks - 1))
        # 维度适配（卷积输出 -> Transformer输入）
        self.proj = nn.Conv1d(conv_out_channels, self.trans_dim, 1)
        # EEG专用Transformer
        self.transformer = EEGTransformer(
            input_dim=self.trans_dim,
            num_heads=self.trans_heads,
            num_layers=self.trans_layers,
            num_classes=self.num_classes,
            seq_len=config.data_loader.segment_length * 128 // (self.max_pooling ** self.num_blocks),
            num_electrodes=self.num_channels
        )

    def forward(self, x):
        # 原卷积+残差块逻辑
        if self.num_channels != 1:
            z = self.mixing_block(x)
        else:
            z = x

        for block, shortcut in zip(self.blocks, self.shortcuts):
            y = shortcut(z)
            z = block(z)
            z += y
            z = self.relu(z)
            z = self.maxpool(z)

        # ===================== Transformer前向 =====================
        z = z.squeeze(2)  # [batch, conv_out, seq_len]
        # 维度投影
        z = self.proj(z)  # [batch, trans_dim, seq_len]
        # Transformer编码+分类
        z = self.transformer(z)
        return z


if __name__ == '__main__':
    import numpy as np
    from src.utils.config import process_config

    config = process_config('./src/configs/test-rnn_model.yaml')
    # 补充Transformer配置（可写入yaml）
    config.network.trans_dim = 64
    config.network.trans_heads = 4
    config.network.trans_layers = 2
    model = EEGTransformerModel(config)
    n_channels = model.num_channels
    length = config.data_loader.segment_length * 128
    x = torch.randn(config.data_loader.batch_size.train, 1, n_channels, length)
    print(model)
    z = model(x)
    print(z.shape)