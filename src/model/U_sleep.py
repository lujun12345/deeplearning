from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.base.base_model import BaseModel

class OutputReshape(nn.Module):
    """对应原Keras的OutputReshape层，处理输出维度转换"""
    def __init__(self, n_periods):
        super().__init__()
        self.n_periods = n_periods

    def forward(self, x):
        # 输入维度: [B, n_classes, seq_len, 1]
        B, C, L, _ = x.shape
        n_pred = L // self.n_periods
        # 重塑为 [B, n_classes, n_periods, n_pred]
        x = x.reshape(B, C, self.n_periods, n_pred)
        if n_pred == 1:
            x = x.squeeze(3)  # 移除n_pred维度
        return x


class PadStartToEvenLength(nn.Module):
    """补齐序列长度为偶数"""
    def forward(self, x):
        L = x.shape[2]  # 序列长度维度
        pad = L % 2
        # PyTorch pad格式: (左, 右, 上, 下) 对应 (W, H) 维度
        x = F.pad(x, (0, 0, pad, 0))  # 仅补齐序列长度维度
        return x

class CropToMatch(nn.Module):
    """裁剪张量以匹配目标维度"""
    def forward(self, inputs):
        x, target = inputs
        x_L = x.shape[2]
        target_L = target.shape[2]
        diff = max(0, x_L - target_L)
        start = diff // 2 + (diff % 2)
        end = start + target_L
        # 裁剪序列长度维度
        x = x[:, :, start:end, :]
        return x


class USleepModel(BaseModel):
    """
    PyTorch实现的USleep模型 (对应NeurIPS 2019论文)
    结构与原TensorFlow版本对齐，格式与rnn_model.py保持一致
    """
    def __init__(self, config):
        super().__init__()
        # 从配置读取核心参数 (需在yaml配置文件中定义)
        self.n_classes = config.data_loader.num_classes #5
        self.n_channels = config.data_loader.num_channels

        self.depth = config.network.get("depth", 12)
        self.dilation = config.network.get("dilation", 1)
        self.activation = config.network.get("activation", "elu")
        self.dense_act = config.network.get("dense_classifier_activation", "tanh")
        self.kernel_size = config.network.get("kernel_size", 9)
        self.transition_window = config.network.get("transition_window", 1)

        self.init_filters = config.network.get("init_filters", 5)
        self.complexity_factor = config.network.get("complexity_factor", 2)

        self.l2_reg = config.network.get("l2_reg")

        self.data_per_prediction = config.network.data_per_prediction # 3840=30*128
        self.n_periods =config.network.n_periods #10
        self.input_dims =config.network.input_dims  # 38400=10*30*128

        if self.input_dims % self.data_per_prediction != 0:
            raise ValueError(f"input_dims({self.input_dims}) 必须被 data_per_prediction({self.data_per_prediction}) 整除")

        # 激活函数映射
        self.act_fn = self._get_activation(self.activation)
        self.dense_act_fn = self._get_activation(self.dense_act)

        # 构建核心模块
        self._build_input_output_layers()
        self._build_encoder()
        self._build_bottom()
        self._build_upsampler()
        self._build_classifier()
        self._init_weights()

    def _get_activation(self, act_name):
        """获取激活函数（与原TF版本对齐）"""
        act_map = {
            "elu": nn.ELU(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "softmax": nn.Softmax(dim=1)
        }
        if act_name not in act_map:
            raise ValueError(f"不支持的激活函数: {act_name}")
        return act_map[act_name]

    def _build_input_output_layers(self):
        """构建输入输出维度转换层"""
        self.pad_start_even = PadStartToEvenLength()
        self.crop_to_match = CropToMatch()
        self.output_reshape = OutputReshape(self.n_periods)

    def _build_encoder(self):
        """构建编码层（下采样）"""
        self.encoder_blocks = nn.ModuleList()
        filters = self.init_filters
        in_channels = self.n_channels
        cf = np.sqrt(self.complexity_factor)

        for i in range(self.depth):
            # 编码块: Conv -> BN -> Activation
            encoder_block = nn.Sequential(OrderedDict([
                (f"enc_conv_{i}", nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=int(filters * cf),
                    kernel_size=(self.kernel_size, 1),
                    padding=(self.kernel_size // 2, 0),  # same padding
                    dilation=(self.dilation, 1)
                )),
                (f"enc_bn_{i}", nn.BatchNorm2d(int(filters * cf))),
            ]))
            self.encoder_blocks.append(encoder_block)

            # 下采样层: MaxPool
            self.encoder_blocks.append(nn.MaxPool2d(kernel_size=(2, 1)))

            # 更新通道数和滤波器数
            in_channels = int(filters * cf)
            filters = int(filters * np.sqrt(2))

        self.encoder_final_in_channels = in_channels
        self.encoder_final_filters = filters

    def _build_bottom(self):
        """构建编码层底部（最深层）"""
        cf = np.sqrt(self.complexity_factor)
        # 计算encoder最后一层的输出通道数
        filters = self.encoder_final_filters   #  self.init_filters * (np.sqrt(2) ** self.depth)
        in_channels = self.encoder_final_in_channels   #  int(self.init_filters * cf * (np.sqrt(2) ** (self.depth - 1)))

        self.bottom_block = nn.Sequential(OrderedDict([
            ("bottom_conv", nn.Conv2d(
                in_channels=in_channels,
                out_channels=int(filters * cf),
                kernel_size=(self.kernel_size, 1),
                padding=(self.kernel_size // 2, 0),
                dilation=(1, 1)
            )),
            ("bottom_bn", nn.BatchNorm2d(int(filters * cf))),
        ]))
        self.bottom_filters = filters

    def _build_upsampler(self):
        """构建解码层（上采样）"""
        self.upsample_blocks = nn.ModuleList()
        cf = np.sqrt(self.complexity_factor)
        filters = self.bottom_filters
        in_channels = int(filters * cf)

        for i in range(self.depth):
            filters = int(np.ceil(filters / np.sqrt(2)))
            # 上采样块: Upsample -> Conv -> BN
            upsample_block = nn.Sequential(OrderedDict([
                (f"up_upsample_{i}", nn.Upsample(scale_factor=(2, 1), mode='nearest')),
                (f"up_conv1_{i}", nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=int(filters * cf),
                    kernel_size=(2, 1),
                    padding=(1, 0)  # same padding for kernel=2
                )),
                (f"up_bn1_{i}", nn.BatchNorm2d(int(filters * cf))),
            ]))
            self.upsample_blocks.append(upsample_block)

            # 拼接后卷积块: Conv -> BN
            concat_conv_block = nn.Sequential(OrderedDict([
                (f"up_conv2_{i}", nn.Conv2d(
                    in_channels=int(filters * cf) * 2,  # 拼接后通道翻倍
                    out_channels=int(filters * cf),
                    kernel_size=(self.kernel_size, 1),
                    padding=(self.kernel_size // 2, 0)
                )),
                (f"up_bn2_{i}", nn.BatchNorm2d(int(filters * cf))),
            ]))
            self.upsample_blocks.append(concat_conv_block)

            in_channels = int(filters * cf)
        self.upsample_final_ch = in_channels

    def _build_classifier(self):
        """构建分类头（密集建模+序列建模）"""
        cf = np.sqrt(self.complexity_factor)
        # 密集建模层
        self.dense_modeling = nn.Conv2d(
            in_channels=self.upsample_final_ch,
            out_channels=int(self.n_classes * cf),
            kernel_size=(1, 1)
        )
        # 序列建模层
        self.avg_pool = nn.AvgPool2d(kernel_size=(self.data_per_prediction, 1))
        self.seq_conv1 = nn.Conv2d(
            in_channels=int(self.n_classes * cf),
            out_channels=self.n_classes,
            kernel_size=(self.transition_window, 1),
            padding=(self.transition_window // 2, 0)
        )
        self.seq_conv2 = nn.Conv2d(
            in_channels=self.n_classes,
            out_channels=self.n_classes,
            kernel_size=(self.transition_window, 1),
            padding=(self.transition_window // 2, 0)
        )

    def _init_weights(self):
        """初始化权重（对齐TF的glorot_uniform + zeros）"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Xavier uniform (glorot_uniform)
                nn.init.xavier_uniform_(module.weight)
                # zeros初始化bias
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """前向传播（严格对齐原TF版本逻辑）"""
        # 1. 输入维度转换
        x = x.squeeze(1).unsqueeze(3) # [B, 1, C, seq_len] -> [B, C, seq_len, 1]
        # 2. 编码阶段
        residual_conns = []
        enc_idx = 0
        for i in range(self.depth):
            # 卷积+BN+激活
            x = self.encoder_blocks[enc_idx](x)
            x = self.act_fn(x)
            # 补齐偶数长度
            x_padded = self.pad_start_even(x)
            # 下采样
            x = self.encoder_blocks[enc_idx + 1](x_padded)
            # 保存残差连接
            residual_conns.append(x_padded)
            enc_idx += 2

        # 3. 编码底部
        x = self.bottom_block(x)
        x = self.act_fn(x)

        # 4. 解码阶段
        residual_conns = residual_conns[::-1]
        up_idx = 0
        for i in range(self.depth):
            # 上采样+卷积+BN+激活
            x = self.upsample_blocks[up_idx](x)
            x = self.act_fn(x)
            # 裁剪以匹配残差连接
            res_conn = residual_conns[i]
            x_cropped = self.crop_to_match([x, res_conn])
            # 拼接残差连接
            x_concat = torch.cat([res_conn, x_cropped], dim=1)
            # 拼接后卷积+BN+激活
            x = self.upsample_blocks[up_idx + 1](x_concat)
            x = self.act_fn(x)
            up_idx += 2

        # 5. 密集建模
        x = self.dense_modeling(x)
        x = self.dense_act_fn(x)

        # 6. 序列建模
        x = self.avg_pool(x)
        x = self.seq_conv1(x)
        x = self.act_fn(x)
        x = self.seq_conv2(x)

        # 7. 输出维度转换
        x = self.output_reshape(x)

        return x

    def compute_loss(self, outputs, y):

        if len(y.shape) == 3 and y.shape[-1] == self.nclass:
            y = torch.argmax(y, dim=-1)  # [batch, epoch_seq_len]

        y = y.long().to(outputs.device)  # 确保设备/类型一致

        output_loss = F.cross_entropy(outputs, y)

        l2_loss = 0.0

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # TF的l2_reg等价于 λ * sum(w²)/2，需除以2
                l2_loss += torch.norm(m.weight, p=2) ** 2 / 2
                if m.bias is not None:
                    l2_loss += torch.norm(m.bias, p=2) ** 2 / 2
        # 4. 总损失
        total_loss = output_loss + self.l2_reg * l2_loss

        return total_loss

