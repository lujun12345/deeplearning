import torch


def adjust_learning_rate_by_weights(optimizer, model, base_lr=1e-4):
    # 取出三个自适应权重
    w1 = model.w1.item()
    w2 = model.w2.item()
    w3 = model.w3.item()

    # 归一化（和loss里使用的一致）
    ws = torch.softmax(torch.tensor([w1, w2, w3]), dim=0)
    w1_norm, w2_norm, w3_norm = ws[0].item(), ws[1].item(), ws[2].item()

    # 核心规则：权重越大 → 学得越好 → lr 越小
    # 最小不低于 base_lr * 0.1
    lr1 = max(base_lr * (1.0 - w1_norm), base_lr * 0.1)
    lr2 = max(base_lr * (1.0 - w2_norm), base_lr * 0.1)
    lr3 = max(base_lr * (1.0 - w3_norm), base_lr * 0.1)

    # 给三个分支设置不同学习率
    for i, param_group in enumerate(optimizer.param_groups):
        if i == 0:  # DeepCNN 分支
            param_group['lr'] = lr1
        elif i == 1:  # SeqSleepNet 分支
            param_group['lr'] = lr2
        elif i == 2:  # Joint 分支
            param_group['lr'] = lr3