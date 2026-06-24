import torch
import torch.nn.functional as F
import numpy as np

class FilteredEEG2TFImage_128Hz(torch.nn.Module):

    """
    针对4阶巴特沃斯滤波后的脑电信号，实现STFT时频转换
    输入：滤波后的时域EEG epoch（30s，采样率128Hz → 3840个采样点）
    输出：128×128时频图（频率轴0~30Hz，时间轴30s）
    """

    def __init__(self,
                 fs=128,               # Sleep-EDF-20采样率固定100Hz
                 epoch_duration=30,    # Epoch时长（秒），固定30s
                 stft_win=128,         # STFT窗长（适配100Hz采样率，更贴合30s时长）
                 stft_hop=64,          # STFT步长（控制时间轴分辨率）
                 n_fft=256,            # FFT点数（决定频率分辨率）
                 freq_min=0,           # 保留最低频率
                 freq_max=50,          # 保留最高频率
                 target_size=(128, 128),
                 norm_mode="channel"): # 输出时频图尺寸
        super().__init__()
        self.fs = fs
        self.epoch_duration = epoch_duration
        self.stft_win = stft_win
        self.stft_hop = stft_hop
        self.n_fft = n_fft
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.target_size = target_size
        self.norm_mode = norm_mode

        # 1. 预计算频率轴，筛选0~30Hz范围
        self.freq_bins = torch.fft.fftfreq(n_fft, d=1/fs)[:n_fft//2 + 1]  # 仅保留正频率
        self.freq_mask = (self.freq_bins >= freq_min) & (self.freq_bins <= freq_max)
        self.valid_freq_num = self.freq_mask.sum()  # 0~30Hz的有效频率点数

        # print("有效频率点：", self.freq_bins[self.freq_mask])
        # print("有效频率数量：", self.valid_freq_num)
        # # 128Hz + n_fft=256 → 0~30Hz 应该正好 61 个点
        # assert self.valid_freq_num == 61, f"0~30Hz有效频率点数量错误：预期61，实际{self.valid_freq_num}"

        # 2. 预定义STFT窗函数（汉宁窗，减少频谱泄漏）
        self.window = torch.hann_window(stft_win, dtype=torch.float32)

    def forward(self, x):

        if self.freq_mask.device != x.device:
            self.freq_mask = self.freq_mask.to(x.device)
            self.freq_bins = self.freq_bins.to(x.device)

        batch_size, epoch_seq_len, nchannel, n_samples = x.shape
        seq_ndim, seq_frame_seq_len = self.target_size

        # 维度重构 → [B*T, C, 3840]（保留epoch_seq_len维度，展平batch+epoch）
        x_reshaped = x.reshape(batch_size * epoch_seq_len * nchannel, n_samples)  # [B*T*C, 3840]

        # --------------------- 步骤1：执行STFT ---------------------
        stft_complex = torch.stft(
            x_reshaped,
            n_fft=self.n_fft,
            win_length=self.stft_win,
            hop_length=self.stft_hop,
            window=self.window.to(x.device),  # 适配GPU/CPU
            return_complex=True,  # 返回复数的实部+虚部（兼容所有PyTorch版本）
            onesided=True  # 仅保留正频率（减少计算量）
        )  # 输出形状：[B*T*C, n_fft//2+1, n_time_frames, 2]
        stft_result = torch.view_as_real(stft_complex)

        # --------------------- 步骤2：计算幅度谱 ---------------------
        stft_magnitude = torch.sqrt(stft_result[..., 0] ** 2 + stft_result[..., 1] ** 2)
        # 筛选0~30Hz的频率分量 → [B*T*C, valid_freq_num, n_time_frames]
        stft_magnitude = stft_magnitude[:, self.freq_mask, :]

        # --------------------- 步骤3：归一化（关键） ---------------------
        eps = torch.finfo(stft_magnitude.dtype).eps  # 动态取数据类型最小精度，避免硬编码1e-8
        if self.norm_mode == "sample":
            # 原逻辑：每个[B*T*C]样本独立归一化（推荐，和原代码效果一致）
            stft_flat = stft_magnitude.reshape(stft_magnitude.shape[0], -1)  # [B*T*C, F*T]
            min_vals = stft_flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)  # [B*T*C, 1, 1]
            max_vals = stft_flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)  # [B*T*C, 1, 1]
            stft_magnitude = (stft_magnitude - min_vals) / (max_vals - min_vals + eps)

        elif self.norm_mode == "channel":
            # 新逻辑：按通道归一化（保留通道间幅值差异）
            # 先拆分通道维度 → [B*T, C, F, T]
            stft_magnitude = stft_magnitude.reshape(
                batch_size * epoch_seq_len,
                nchannel,
                self.valid_freq_num,
                stft_magnitude.shape[-1]
            )  # [B*T, C, F, T]
            # 按通道维度归一化（dim=-1：对每个通道的所有频率+时间点取min/max）
            stft_flat = stft_magnitude.reshape(batch_size * epoch_seq_len, nchannel, -1)  # [B*T, C, F*T]
            min_vals = stft_flat.min(dim=-1, keepdim=True)[0].unsqueeze(-1)  # [B*T, C, 1, 1]
            max_vals = stft_flat.max(dim=-1, keepdim=True)[0].unsqueeze(-1)  # [B*T, C, 1, 1]
            stft_magnitude = (stft_magnitude - min_vals) / (max_vals - min_vals + eps)
        else:
            raise ValueError(f"归一化模式{self.norm_mode}不支持，可选：sample/channel")
        # --------------------- 步骤4：调整尺寸到128×128 ---------------------
        # 分离出通道维度 → [B*T, C, valid_freq_num, n_time_frames]

        current_freq_dim = self.valid_freq_num
        current_time_dim = stft_magnitude.shape[-1]
        stft_magnitude = stft_magnitude.reshape(
            batch_size * epoch_seq_len,
            nchannel,
            current_freq_dim,
            current_time_dim
        )
        # 双线性插值到目标尺寸（对齐MATLAB imresize逻辑）
        tf_image_btc = F.interpolate(
            stft_magnitude,
            size=self.target_size,
            mode='bilinear',
            align_corners=False
        )  # 输出形状：[B*T, C, seq_ndim, seq_frame_seq_len]

        # 转置为 [B*T, seq_frame_seq_len, seq_ndim, C]
        tf_image_btc = tf_image_btc.transpose(1, 3)
        # 最终还原为 [B, T, seq_frame_seq_len, seq_ndim, C]
        tf_image = tf_image_btc.reshape(batch_size, epoch_seq_len, seq_frame_seq_len, seq_ndim, nchannel)

        return tf_image

if __name__ == "__main__":
    # 模拟：4阶巴特沃斯滤波后的128Hz×30s EEG数据（batch=8，1通道，3840采样点）
    batch_filtered_eeg = torch.randn(32, 10, 5, 3840, dtype=torch.float32)  # 3840=128×30

    # 初始化转换器（128Hz参数）
    tf_converter = FilteredEEG2TFImage_128Hz(
        fs=128,
        stft_win=128,
        stft_hop=64,
        n_fft=256,
        freq_max=30,
        target_size=(128, 60)
    )

    # 转换并验证
    tf_images = tf_converter(batch_filtered_eeg)
    print(f"输入滤波后EEG形状：{batch_filtered_eeg.shape}")  # [32,10,5,3840]
    print(f"输出时频图形状：{tf_images.shape}")              # [32,10,128,128,5]
    print(f"时频图数值范围：[{tf_images.min().item():.4f}, {tf_images.max().item():.4f}]")  # [0,1]