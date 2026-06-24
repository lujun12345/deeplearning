import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from src.model.pre_x2 import FilteredEEG2TFImage_128Hz

class BidirectionalRNN(nn.Module):
    """双向RNN封装（支持BatchNorm）"""

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.,
                 batch_norm=True, rnn_type= "GRU" ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.rnn_type = rnn_type

        # 定义双向GRU（原代码未指定RNN类型，默认用GRU，可替换为LSTM）
        if rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
        # self.rnn.flatten_parameters()

        # BatchNorm层（匹配原代码的batch norm逻辑）
        if batch_norm:
            self.bn = nn.BatchNorm1d(2 * hidden_size)

    def forward(self, x):
        # 处理变长序列（可选，简化版暂不实现pack_padded_sequence）
        out, _ = self.rnn(x)
        if self.batch_norm:
            # BatchNorm需要 [batch, features, seq_len] 格式
            out = out.transpose(1, 2)
            out = self.bn(out)
            out = out.transpose(1, 2)
        return out


class AttentionLayer(nn.Module):
    """注意力层（匹配原代码的attention逻辑）"""

    def __init__(self, input_size, attention_size):
        super().__init__()
        self.attention_size = attention_size
        self.W = nn.Linear(input_size, attention_size)
        self.v = nn.Linear(attention_size, 1, bias=False)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        score = torch.tanh(self.W(x))  # [batch, seq_len, attention_size]
        attention_weights = F.softmax(self.v(score), dim=1)  # [batch, seq_len, 1]
        context = x * attention_weights  # [batch, seq_len, input_size]
        context = torch.sum(context, dim=1)  # [batch, input_size]
        return context, attention_weights


class DynamicConv1dBlock(nn.Module):
    """1D动态卷积块（适配EEG信号，替换原静态DownConvBlock）"""

    def __init__(self, in_channels, out_channels, kernel_size=31, stride=2, padding=15,
                 reduction=4 ):  # reduction：生成卷积核的MLP维度缩减系数
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 1. 全局池化：提取输入特征的全局信息（适配EEG序列）
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # 2. MLP生成动态卷积核参数（缩减维度降低计算量）
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.LeakyReLU(0.2),
            nn.Linear(in_channels // reduction, out_channels * in_channels * kernel_size)
        )

        # 3. 静态偏置（可选，也可动态生成）
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # 4. 激活函数（对齐原代码）
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        # x: [batch, in_channels, seq_len]
        batch_size, in_channels, seq_len = x.shape

        # 步骤1：全局池化提取全局特征 [batch, in_channels, 1] -> [batch, in_channels]
        global_feat = self.global_avg_pool(x).squeeze(-1)

        # 步骤2：MLP生成动态卷积核 [batch, out_channels*in_channels*kernel_size]
        dynamic_kernel = self.mlp(global_feat)
        # 重塑卷积核形状 [batch, out_channels, in_channels, kernel_size]
        dynamic_kernel = dynamic_kernel.reshape(
            batch_size, self.out_channels, self.in_channels, self.kernel_size
        )

        # 步骤3：执行动态卷积（利用group conv实现逐样本卷积）
        # 调整输入形状 [batch, in_channels, seq_len] -> [1, batch*in_channels, seq_len]
        x_input = x.reshape(1, batch_size * in_channels, seq_len)
        # 调整卷积核形状 [batch*out_channels, in_channels, kernel_size]
        dynamic_kernel = dynamic_kernel.reshape(
            batch_size * self.out_channels, self.in_channels, self.kernel_size
        )
        # 分组卷积（group=batch_size，实现逐样本动态卷积）
        out = F.conv1d(
            x_input,
            weight=dynamic_kernel,
            bias=self.bias.repeat(batch_size),  # 偏置扩展到batch维度
            stride=self.stride,
            padding=self.padding,
            groups=batch_size  # 关键：每个样本对应一个group
        )

        # 步骤4：恢复形状 + 激活
        out = out.reshape(batch_size, self.out_channels, -1)
        out = self.leaky_relu(out)

        return out

class DownConvBlock(nn.Module):
    """下采样卷积块（匹配原代码的downconv+leakyrelu）"""

    def __init__(self, in_channels, out_channels, kernel_size=31, stride=2, padding=15):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.leaky_relu(x)
        return x

class XSleepNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        # self.to(torch.device('cuda:0'))
        self.epoch_seq_len = config.network.epoch_seq_len  #10

        self.nchannel = config.data_loader.num_channels
        self.nclass = config.data_loader.num_classes
        self.deep_ntime= config.network.deep_ntime # 3840=30*128

        self.l2_reg_lambda = config.network.l2_reg_lambda #0.001

        # DeepCNN分支参数
        self.deep_nhidden = config.network.deep_nhidden #256
        self.deep_nlayer = config.network.deep_nlayer #2
        self.g_enc_depths = config.network.g_enc_depths  # CNN深度 [16, 16, 32, 32, 64, 64, 128, 128, 256]

        # SeqSleepNet分支参数
        self.seq_ndim = config.network.seq_ndim  #128
        self.seq_frame_seq_len = config.network.seq_frame_seq_len  #60
        self.seq_nfilter = config.network.seq_nfilter  # 32

        self.seq_nhidden1 = config.network.seq_nhidden1  #64
        self.seq_nlayer1 = config.network.seq_nlayer1  #1
        self.seq_attention_size1 = config.network.seq_attention_size1  #64
        self.seq_nhidden2 = config.network.seq_nhidden2  #256
        self.seq_nlayer2 = config.network.seq_nlayer2  #1

        # Dropout参数
        self.dropout_rnn = config.network.dropout_rnn  #0.25
        self.dropout_cnn = config.network.dropout_cnn  #0.5

        self.tf_converter = FilteredEEG2TFImage_128Hz(
            fs=128, stft_win=128, stft_hop=64, n_fft=256, freq_max=30,
            target_size=(self.seq_ndim, self.seq_frame_seq_len)
        )

        # self.class_counts= config.data_loader.class_counts

        # if self.class_counts is not None:
        #     # 传入每个类的样本数 → 自动计算权重（解决类别不平衡）
        #     class_counts = np.array(self.class_counts, dtype=np.float32)
        #     # 标准权重公式：总样本数 / (类别数 × 该类样本数)
        #     class_weights = class_counts.sum() / (len(class_counts) * class_counts)
        #     # 归一化（可选，确保最小权重为1，避免权重差异过大）
        #     class_weights = class_weights / class_weights.min()
        #     # 转为torch tensor，绑定到模型（自动同步设备）
        #     self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        # else:
        #     self.class_weights = None  # 无权重时用普通交叉熵

        # 2. 构建DeepCNN分支（对应construct_all_cnn_net）
        self._build_deep_cnn_branch()

        # 3. 构建SeqSleepNet分支（对应construct_seqsleepnet）
        self._build_seq_sleepnet_branch()

        # 4. 构建Joint分支（对应construct_joint_branch）
        self._build_joint_branch()

        # 5. 权重参数（原代码的w1/w2/w3，训练时可优化）
        # self.w1 = nn.Parameter(torch.ones(1))
        # self.w2 = nn.Parameter(torch.ones(1))
        # self.w3 = nn.Parameter(torch.ones(1))

        #初始化参数
        self._init_weights()

    def _build_deep_cnn_branch(self):
        """构建DeepCNN+双向RNN分支"""
        # 构建下采样卷积链
        in_channels = self.nchannel
        self.deep_cnn_blocks = nn.ModuleList()
        # 构建下采样卷积链
        for idx, depth in enumerate(self.g_enc_depths[:-1]):  # 除了最后1层，都用普通卷积
            self.deep_cnn_blocks.append(
                DownConvBlock(
                    in_channels=in_channels,
                    out_channels=depth,
                    kernel_size=31,
                    stride=2,
                    padding=15
                )
            )
            in_channels = depth

        for idx, depth in enumerate(self.g_enc_depths[-1:]):
            self.deep_cnn_blocks.append(
                DynamicConv1dBlock(  # 替换原DownConvBlock
                    in_channels=in_channels,
                    out_channels=depth,
                    kernel_size=15,
                    stride=2,
                    padding=7,
                    reduction=4  # 可根据需求调整维度缩减系数
                )
            )
            in_channels = depth


        cnn_time_steps = self.deep_ntime
        for idx, depth in enumerate(self.g_enc_depths):
            if idx < len(self.g_enc_depths) - 1:  # 普通卷积
                kernel = 31
                padding = 15
            else:  # 动态卷积
                kernel = 15
                padding = 7
            cnn_time_steps = (cnn_time_steps + 2 * padding - kernel) // 2 + 1
        # for _ in self.g_enc_depths:
        #     cnn_time_steps = (cnn_time_steps + 2 * 15 - 31) // 2 + 1  # 卷积下采样公式：(W+2P-K)/S +1

        self._deep_rnn_input_size = cnn_time_steps * self.g_enc_depths[-1]

        # 双向RNN
        self.deep_rnn = BidirectionalRNN(
            input_size=self._deep_rnn_input_size,
            hidden_size=self.deep_nhidden,
            num_layers=self.deep_nlayer,
            dropout=self.dropout_rnn,
            batch_norm=False
        )

        # Deep分支输出层（每个epoch一个分类头）
        self.deep_output_layers = nn.ModuleList([
            nn.Linear(2 * self.deep_nhidden, self.nclass)
            for _ in range(self.epoch_seq_len)
        ])

    def _build_seq_sleepnet_branch(self):
        """构建SeqSleepNet分支（滤波器组+双向RNN+注意力）"""

        # 2. 通道级滤波器权重（EEG/EOG/EMG）
        if self.nchannel == 4 :
            self.Weeg1 = nn.Parameter(torch.randn(self.seq_ndim, self.seq_nfilter))
            self.Weeg2 = nn.Parameter(torch.randn(self.seq_ndim, self.seq_nfilter))
            self.Weog = nn.Parameter(torch.randn(self.seq_ndim, self.seq_nfilter))
            self.Wemg = nn.Parameter(torch.randn(self.seq_ndim, self.seq_nfilter))

        elif self.nchannel == 5 :
            self.Weeg1 = nn.Parameter(torch.randn(self.seq_ndim, self.seq_nfilter))
            self.Weeg2 = nn.Parameter(torch.randn(self.seq_ndim, self.seq_nfilter))
            self.Weog1 = nn.Parameter(torch.randn(self.seq_ndim, self.seq_nfilter))
            self.Weog2 = nn.Parameter(torch.randn(self.seq_ndim, self.seq_nfilter))
            self.Wemg = nn.Parameter(torch.randn(self.seq_ndim, self.seq_nfilter))

        elif self.nchannel == 3 :
            self.Weeg = nn.Parameter(torch.randn(self.seq_ndim, self.seq_nfilter))
            self.Weog = nn.Parameter(torch.randn(self.seq_ndim, self.seq_nfilter))
            self.Wemg = nn.Parameter(torch.randn(self.seq_ndim, self.seq_nfilter))

        else:
            raise ValueError(f"Unsupported nchannel: {self.nchannel}, only 3/4/5 are supported")

        # 3. 帧级双向RNN（frame-level）
        self.seq_frame_rnn = BidirectionalRNN(
            input_size=self.seq_nfilter * self.nchannel,
            hidden_size=self.seq_nhidden1,
            num_layers=self.seq_nlayer1,
            dropout=self.dropout_rnn,
            batch_norm=True,
            rnn_type="LSTM"    # 原代码seq分支有BN
        )

        # 4. 注意力层（frame-level）
        self.seq_attention = AttentionLayer(
            input_size = 2 * self.seq_nhidden1,
            attention_size = self.seq_attention_size1
        )

        # 5. 序列级双向RNN（epoch-level）
        self.seq_epoch_rnn = BidirectionalRNN(
            input_size=2 * self.seq_nhidden1,
            hidden_size=self.seq_nhidden2,
            num_layers=self.seq_nlayer2,
            dropout=self.dropout_rnn,
            batch_norm=True,
            rnn_type="LSTM"
        )

        # 6. Seq分支输出层（每个epoch一个分类头）
        self.seq_output_layers = nn.ModuleList([
            nn.Linear(2 * self.seq_nhidden2, self.nclass)
            for _ in range(self.epoch_seq_len)
        ])

    def _build_joint_branch(self):
        """构建Joint分支（Deep+Seq特征融合）"""
        self.joint_output_layers = nn.ModuleList([
            nn.Linear(2 * self.deep_nhidden + 2 * self.seq_nhidden2, self.nclass)
            for _ in range(self.epoch_seq_len)
        ])

    def _init_weights(self):
        """初始化层权重，对齐原TF逻辑（新增动态卷积初始化）"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv1d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            # 新增：动态卷积的MLP和偏置初始化
            elif isinstance(m, DynamicConv1dBlock):
                # MLP层初始化
                for mlp_layer in m.mlp:
                    if isinstance(mlp_layer, nn.Linear):
                        nn.init.trunc_normal_(mlp_layer.weight, std=0.02)
                        if mlp_layer.bias is not None:
                            nn.init.constant_(mlp_layer.bias, 0.0)
                # 动态卷积偏置初始化
                nn.init.constant_(m.bias, 0.0)

    # def _init_weights(self):
    #     """初始化层权重，对齐原TF逻辑"""
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.trunc_normal_(m.weight, std=0.02)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0.0)
    #         elif isinstance(m, nn.Conv1d):
    #             nn.init.trunc_normal_(m.weight, std=0.02)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0.0)
    #         elif isinstance(m, nn.BatchNorm1d):
    #             nn.init.constant_(m.weight, 1.0)
    #             nn.init.constant_(m.bias, 0.0)

    def _deep_cnn_forward(self, x1, training=True):

        batch_size = x1.shape[0]
        # Reshape: [batch,epoch_seq_len, nchannel, deep_ntime] -> [batch*epoch_seq_len, nchannel, deep_ntime]
        x1 = x1.flatten(0, 1)
        """DeepCNN分支前向传播"""
        # 执行所有下采样卷积
        for conv_block in self.deep_cnn_blocks:
            x1 = conv_block(x1)
            if self.training:
                x1 = F.dropout(x1, p=self.dropout_cnn)

        cnn_time_steps = x1.shape[-1]
        assert cnn_time_steps * self.g_enc_depths[-1] == self._deep_rnn_input_size,\
            f"Deep RNN input size mismatch: expected {self._deep_rnn_input_size}, got {cnn_time_steps * self.g_enc_depths[-1]}"

        # Reshape: 匹配原代码的维度变换
        x1 = x1.reshape(batch_size, self.epoch_seq_len, -1)   # [batch, epoch_seq_len, cnn_time_steps * self.g_enc_depths[-1]]

        deep_rnn_out = self.deep_rnn(x1)  # [batch, epoch_seq_len, 2*deep_nhidden]

        # 每个epoch的输出
        deep_scores = []
        for i in range(self.epoch_seq_len):
            score_i = self.deep_output_layers[i](deep_rnn_out[:, i, :]) #[batch, nclass]
            deep_scores.append(score_i)

        return deep_scores, deep_rnn_out

    def _seq_sleepnet_forward(self, x2):
        """SeqSleepNet分支前向传播"""
        # x2: [batch, epoch_seq_len, seq_frame_seq_len, seq_ndim, nchannel]
        batch_size, epoch_seq_len, seq_frame_seq_len, seq_ndim, nchannel = x2.shape[0],x2.shape[1],x2.shape[2],x2.shape[3],x2.shape[4]

        # 1. 滤波器组处理（EEG/EOG/EMG通道）

        if self.nchannel == 5:
            # 提取EEG通道
            xeeg1 = x2[:, :, :, :, 0].reshape(-1, self.seq_ndim)  # [batch*epoch_seq_len*seq_frame_seq_len, seq_ndim]
            Weeg1 = torch.sigmoid(self.Weeg1)  # 非负约束
            # Wfb_eeg = Weeg * self.Wbl.to(xeeg.device)  # 滤波器掩码
            hweeg1 = torch.matmul(xeeg1, Weeg1)  # [batch*epoch_seq_len*seq_frame_seq_len, seq_nfilter]
            hweeg1 = hweeg1.reshape(batch_size, self.epoch_seq_len, self.seq_frame_seq_len, self.seq_nfilter)

            xeeg2 = x2[:, :, :, :, 1].reshape(-1, self.seq_ndim)  # [batch*epoch_seq_len*seq_frame_seq_len, seq_ndim]
            Weeg2 = torch.sigmoid(self.Weeg2)  # 非负约束
            # Wfb_eeg = Weeg * self.Wbl.to(xeeg.device)  # 滤波器掩码
            hweeg2 = torch.matmul(xeeg2, Weeg2)  # [batch*epoch_seq_len*seq_frame_seq_len, seq_nfilter]
            hweeg2 = hweeg2.reshape(batch_size, self.epoch_seq_len, self.seq_frame_seq_len, self.seq_nfilter)

            # 提取EOG通道（如果有）
            xeog1 = x2[:, :, :, :, 2].reshape(-1, self.seq_ndim)
            Weog1 = torch.sigmoid(self.Weog1)
            # Wfb_eog = Weog * self.Wbl.to(xeog.device)
            hweog1 = torch.matmul(xeog1, Weog1)
            hweog1 = hweog1.reshape(batch_size, self.epoch_seq_len, self.seq_frame_seq_len, self.seq_nfilter)

            xeog2 = x2[:, :, :, :, 3].reshape(-1, self.seq_ndim)
            Weog2 = torch.sigmoid(self.Weog2)
            # Wfb_eog = Weog * self.Wbl.to(xeog.device)
            hweog2 = torch.matmul(xeog2, Weog2)
            hweog2 = hweog2.reshape(batch_size, self.epoch_seq_len, self.seq_frame_seq_len, self.seq_nfilter)

            # 提取EMG通道（如果有）
            xemg = x2[:, :, :, :, 4].reshape(-1, self.seq_ndim)
            Wemg = torch.sigmoid(self.Wemg)
            # Wfb_emg = Wemg * self.Wbl.to(xemg.device)
            hwemg = torch.matmul(xemg, Wemg)
            hwemg = hwemg.reshape(batch_size, self.epoch_seq_len, self.seq_frame_seq_len, self.seq_nfilter)

            # 拼接多通道
            x_concat = torch.cat([hweeg1, hweeg2, hweog1, hweog2, hwemg], dim=-1)

        elif self.nchannel == 4 :
            # 提取EEG通道
            xeeg1 = x2[:, :, :, :, 0].reshape(-1, self.seq_ndim)  # [batch*epoch_seq_len*seq_frame_seq_len, seq_ndim]
            Weeg1 = torch.sigmoid(self.Weeg1)  # 非负约束
            # Wfb_eeg = Weeg * self.Wbl.to(xeeg.device)  # 滤波器掩码
            hweeg1 = torch.matmul(xeeg1, Weeg1)  # [batch*epoch_seq_len*seq_frame_seq_len, seq_nfilter]
            hweeg1 = hweeg1.reshape(batch_size, self.epoch_seq_len, self.seq_frame_seq_len, self.seq_nfilter)

            xeeg2 = x2[:, :, :, :, 1].reshape(-1, self.seq_ndim)  # [batch*epoch_seq_len*seq_frame_seq_len, seq_ndim]
            Weeg2 = torch.sigmoid(self.Weeg2)  # 非负约束
            # Wfb_eeg = Weeg * self.Wbl.to(xeeg.device)  # 滤波器掩码
            hweeg2 = torch.matmul(xeeg2, Weeg2)  # [batch*epoch_seq_len*seq_frame_seq_len, seq_nfilter]
            hweeg2 = hweeg2.reshape(batch_size, self.epoch_seq_len, self.seq_frame_seq_len, self.seq_nfilter)

            # 提取EOG通道
            xeog = x2[:, :, :, :, 2].reshape(-1, self.seq_ndim)
            Weog = torch.sigmoid(self.Weog)
            # Wfb_eog = Weog * self.Wbl.to(xeog.device)
            hweog = torch.matmul(xeog, Weog)
            hweog = hweog.reshape(batch_size, self.epoch_seq_len, self.seq_frame_seq_len, self.seq_nfilter)

            # 提取EMG通道
            xemg = x2[:, :, :, :, 3].reshape(-1, self.seq_ndim)
            Wemg = torch.sigmoid(self.Wemg)
            # Wfb_emg = Wemg * self.Wbl.to(xemg.device)
            hwemg = torch.matmul(xemg, Wemg)
            hwemg = hwemg.reshape(batch_size, self.epoch_seq_len, self.seq_frame_seq_len, self.seq_nfilter)

            # 拼接多通道
            x_concat = torch.cat([hweeg1, hweeg2, hweog, hwemg], dim=-1)

        elif self.nchannel == 3:
            # 提取EEG通道
            xeeg = x2[:, :, :, :, 0].reshape(-1, self.seq_ndim)  # [batch*epoch_seq_len*seq_frame_seq_len, seq_ndim]
            Weeg = torch.sigmoid(self.Weeg)  # 非负约束
            # Wfb_eeg = Weeg * self.Wbl.to(xeeg.device)  # 滤波器掩码
            hweeg = torch.matmul(xeeg, Weeg)  # [batch*epoch_seq_len*seq_frame_seq_len, seq_nfilter]
            hweeg = hweeg.reshape(batch_size, self.epoch_seq_len, self.seq_frame_seq_len, self.seq_nfilter)

            # 提取EOG通道（如果有）
            xeog = x2[:, :, :, :, 1].reshape(-1, self.seq_ndim)
            Weog = torch.sigmoid(self.Weog)
            # Wfb_eog = Weog * self.Wbl.to(xeog.device)
            hweog = torch.matmul(xeog, Weog)
            hweog = hweog.reshape(batch_size, self.epoch_seq_len, self.seq_frame_seq_len, self.seq_nfilter)

            # 提取EMG通道（如果有）
            xemg = x2[:, :, :, :, 2].reshape(-1, self.seq_ndim)
            Wemg = torch.sigmoid(self.Wemg)
            # Wfb_emg = Wemg * self.Wbl.to(xemg.device)
            hwemg = torch.matmul(xemg, Wemg)
            hwemg = hwemg.reshape(batch_size, self.epoch_seq_len, self.seq_frame_seq_len, self.seq_nfilter)

            # 拼接多通道
            x_concat = torch.cat([hweeg, hweog, hwemg], dim=-1)

        else:
            raise ValueError(f"Unsupported nchannel: {self.nchannel}, only 3/4/5 are supported")

        # 1. Reshape: [batch*epoch_seq_len, seq_frame_seq_len, seq_nfilter*nchannel]
        x = x_concat.reshape(-1, self.seq_frame_seq_len, self.seq_nfilter * self.nchannel)

        # 2. 帧级双向RNN
        seq_frame_rnn_out = self.seq_frame_rnn(x)  # [batch*epoch_seq_len, seq_frame_seq_len, 2*seq_nhidden1]

        # 3. 注意力层
        attention_out, attn_weights = self.seq_attention(seq_frame_rnn_out) # [batch*epoch_seq_len, 2*seq_nhidden1]

        # 4. reshape
        attention_out = attention_out.reshape(batch_size, self.epoch_seq_len, 2 * self.seq_nhidden1)

        # 5. epoch级双向RNN
        seq_epoch_rnn_out = self.seq_epoch_rnn(attention_out) # [batch*epoch_seq_len, seq_frame_seq_len, 2*seq_nhidden2]

        # 6. 每个epoch的输出
        seq_scores = []
        for i in range(self.epoch_seq_len):
            score_i = self.seq_output_layers[i](seq_epoch_rnn_out[:, i, :])  # [batch, nclass]
            seq_scores.append(score_i)

        return seq_scores, seq_epoch_rnn_out

    def _joint_forward(self, deep_rnn_out, seq_rnn_out):
        """Joint分支前向传播"""
        joint_scores = []
        for i in range(self.epoch_seq_len):
            # 拼接Deep和Seq特征
            feat = torch.cat([
                deep_rnn_out[:, i, :],  # [batch, 2*deep_nhidden]
                seq_rnn_out[:, i, :]  # [batch, 2*seq_nhidden2]
            ], dim=1)
            score_i = self.joint_output_layers[i](feat)  # [batch, nclass]
            joint_scores.append(score_i)
        return joint_scores

    def forward(self, x):
        """
        整体前向传播
        :param x1: DeepCNN分支输入, shape=[batch, epoch_seq_len, deep_ntime, nchannel]
        :param x2: SeqSleepNet分支输入, shape=[batch, epoch_seq_len, seq_frame_seq_len, seq_ndim, nchannel]
        :return: 融合后的预测分数、各分支分数
        """
        # 1. 各分支前向传播
        x1= x
        # x1的形状为[batchsize,1,nchannel,epoch_seq_len* deep_ntime]

        batch_size = x1.shape[0]
        x1_squeezed = x1.squeeze(1)

        assert x1_squeezed.shape[-1] % self.epoch_seq_len == 0, "x1长度需被epoch_seq_len整除"
        deep_ntime = x1_squeezed.shape[-1] // self.epoch_seq_len
        x1_reshaped = x1_squeezed.reshape(batch_size, self.nchannel, self.epoch_seq_len, deep_ntime)
        x1 = x1_reshaped.transpose(1, 2)  # 交换第2维和第3维，得到 [32,10,5,3840]

        x2= self.tf_converter(x1)

        deep_scores, deep_rnn_out = self._deep_cnn_forward(x1)
        seq_scores, seq_rnn_out = self._seq_sleepnet_forward(x2)
        joint_scores = self._joint_forward(deep_rnn_out, seq_rnn_out)

        # # 2. 融合分数（匹配原代码的权重加权）
        fused_scores = []
        for i in range(self.epoch_seq_len):
            # 权重归一化（避免权重和不为1）
            # weights = torch.cat([self.w1, self.w2, self.w3], dim=0)
            # w1_norm, w2_norm, w3_norm = torch.softmax(weights, dim=0)
            fused_score =  (deep_scores[i] + seq_scores[i] +  joint_scores[i]) / 3

            fused_scores.append(fused_score)

        fused_scores = torch.stack(fused_scores, dim=1).transpose(1,2)

        # 3. 返回结果（按需调整，可返回各分支分数用于计算损失）
        return {
            "fused_scores": fused_scores,
            "deep_scores": deep_scores,
            "seq_scores": seq_scores,
            "joint_scores": joint_scores
        }

    def compute_loss(self, outputs, y):
        """
        计算总损失（匹配原代码的loss逻辑）
        :param outputs: forward输出的字典
        :param y: 标签, shape=[batch, epoch_seq_len, nclass]
        :return: 总损失、各分支损失
        """
        deep_scores = outputs["deep_scores"]
        seq_scores = outputs["seq_scores"]
        joint_scores = outputs["joint_scores"]

        if len(y.shape) == 3 and y.shape[-1] == self.nclass:
            y = torch.argmax(y, dim=-1)  # [batch, epoch_seq_len]

        y = y.long().to(deep_scores[0].device)  # 确保设备/类型一致

        # class_weights = self.class_weights.to(deep_scores[0].device) \
        #     if self.class_weights is not None else None

        # 1. 计算各分支的交叉熵损失
        deep_loss = 0.
        seq_loss = 0.
        joint_loss = 0.
        for i in range(self.epoch_seq_len):
            # 标签: [batch, nclass]
            y_i = y[:, i]
            # Deep分支损失
            deep_loss_i = F.cross_entropy(deep_scores[i], y_i)
            deep_loss += deep_loss_i
            # Seq分支损失
            seq_loss_i = F.cross_entropy(seq_scores[i], y_i)
            seq_loss += seq_loss_i
            # Joint分支损失
            joint_loss_i = F.cross_entropy(joint_scores[i], y_i)
            joint_loss += joint_loss_i

        # 平均到每个epoch
        deep_loss /= self.epoch_seq_len
        seq_loss /= self.epoch_seq_len
        joint_loss /= self.epoch_seq_len

        # weights = torch.cat([self.w1, self.w2, self.w3], dim=0)
        # w1_norm, w2_norm, w3_norm = torch.softmax(weights, dim=0)

        output_loss = (deep_loss +  seq_loss +  joint_loss) / 3

        # 3. L2正则化（排除滤波器组层）
        l2_loss = 0.
        for name, param in self.named_parameters():
            # 排除滤波器组层的参数
            if "Weeg" not in name and "Weog" not in name and "Wemg" not in name:
                l2_loss += 0.5 * torch.norm(param, p=2) ** 2

        # 4. 总损失
        total_loss = output_loss + self.l2_reg_lambda * l2_loss

        return total_loss