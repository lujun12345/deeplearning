import torch
import numpy as np
from scipy import stats


class AdaptiveWeightScheduler:
    """自适应权重调度器（实现论文 XSleepNet2 逻辑）"""
    def __init__(self, num_branches=3, window_size=20, epsilon=1e-8):
        self.num_branches = num_branches  # Deep(0), Seq(1), Joint(2)
        self.window_size = window_size  # 线拟合窗口（论文设为20）
        self.epsilon = epsilon  # 避免除以零

        # 记录各分支的训练损失和验证损失（历史序列）
        self.train_loss_history = [[] for _ in range(num_branches)]
        self.val_loss_history = [[] for _ in range(num_branches)]

        # 参考点 n0（初始为0，后续更新为验证损失最优的步骤）
        self.n0 = 0
        # 参考点对应的损失斜率（train/val）
        self.ref_train_slope = [0.0 for _ in range(num_branches)]
        self.ref_val_slope = [0.0 for _ in range(num_branches)]

    def _fit_slope(self, loss_history):
        """用线性回归拟合损失序列的斜率（近似切线）"""
        if len(loss_history) < self.window_size:
            # 热身期：返回0（权重均等）
            return 0.0
        # 取最近 window_size 个损失值
        recent_losses = np.array(loss_history[-self.window_size:])
        steps = np.arange(len(recent_losses))
        # 线性回归求斜率
        slope, _, _, _, _ = stats.linregress(steps, recent_losses)
        return slope

    def update_losses(self, train_losses, val_losses):
        """更新各分支的训练/验证损失历史
        Args:
            train_losses: 列表 [deep_train_loss, seq_train_loss, joint_train_loss]
            val_losses: 列表 [deep_val_loss, seq_val_loss, joint_val_loss]
        """
        for k in range(self.num_branches):
            self.train_loss_history[k].append(train_losses[k].item())
            self.val_loss_history[k].append(val_losses[k].item())

        # 检查是否需要更新参考点 n0（当任意分支验证损失达到历史最优）
        current_step = len(self.val_loss_history[0]) - 1
        if current_step >= self.window_size:
            for k in range(self.num_branches):
                # 取当前窗口的验证损失均值
                current_val_mean = np.mean(self.val_loss_history[k][-self.window_size:])
                # 取参考点窗口的验证损失均值
                ref_val_mean = np.mean(self.val_loss_history[k][self.n0:self.n0 + self.window_size])
                # 若当前更优，更新参考点
                if current_val_mean < ref_val_mean - self.epsilon:
                    self.n0 = current_step
                    # 更新参考点的斜率
                    self.ref_train_slope[k] = self._fit_slope(
                        self.train_loss_history[k][self.n0:self.n0 + self.window_size])
                    self.ref_val_slope[k] = self._fit_slope(
                        self.val_loss_history[k][self.n0:self.n0 + self.window_size])

    def compute_weights(self):
        """计算自适应权重 w1/w2/w3（对应 Deep/Seq/Joint）"""
        weights = []
        for k in range(self.num_branches):
            # 1. 计算当前损失斜率（train/val）
            current_train_slope = self._fit_slope(self.train_loss_history[k])
            current_val_slope = self._fit_slope(self.val_loss_history[k])

            # 2. 计算泛化度量 G_k = 当前val斜率 - 参考点val斜率（负斜率表示泛化，G_k为负→取绝对值）
            G_k = current_val_slope - self.ref_val_slope[k]
            G_k = max(-G_k, self.epsilon)  # 确保G_k非负（泛化越好，G_k越大）

            # 3. 计算过拟合度量 O_k = (当前val斜率 - 当前train斜率) - (参考点val斜率 - 参考点train斜率)
            O_k = (current_val_slope - current_train_slope) - (self.ref_val_slope[k] - self.ref_train_slope[k])
            O_k = abs(O_k) + self.epsilon  # 确保O_k非负（过拟合越轻，O_k越小）

            # 4. 计算分支权重
            w_k = G_k / (O_k ** 2)
            weights.append(w_k)

        # 5. 归一化（Z = sum(weights)）
        weights = np.array(weights)
        weights = weights / (weights.sum() + self.epsilon)
        return {
            "w1": torch.tensor(weights[0], dtype=torch.float32),  # Deep
            "w2": torch.tensor(weights[1], dtype=torch.float32),  # Seq
            "w3": torch.tensor(weights[2], dtype=torch.float32)  # Joint
        }