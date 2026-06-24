import torch
print(f"PyTorch 版本：{torch.__version__}")
print(f"CUDA 是否可用：{torch.cuda.is_available()}")
print(f"GPU 名称：{torch.cuda.get_device_name(0)}")
print(f"支持的算力：{torch.cuda.get_arch_list()}")