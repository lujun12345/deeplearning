import copy
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
import math


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

    
class Window_Embedding(nn.Module): 
    def __init__(self, config):
        super().__init__()
        self.emb_size=config.network.d_model
        self.window_size=config.network.window_size       #64
        self.in_channels=config.network.num_channels    #1

        self.projection_1 =  nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains, in=>B,1,3000 out=>B,64,60
            nn.Conv1d(self.in_channels, self.emb_size//4, kernel_size = self.window_size, stride = self.window_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.emb_size//4)
            # Rearrange('b e s -> b s e'),
            )
        self.projection_2 =  nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains, in=>B,1,3000 out=>B,64,60
            nn.Conv1d(self.in_channels, self.emb_size//8, kernel_size = 8, stride = 8),
            nn.LeakyReLU(),
            nn.Conv1d(self.emb_size//8, self.emb_size//4, kernel_size = 4, stride = 4),
            nn.LeakyReLU(),
            nn.Conv1d(self.emb_size//4, (self.emb_size-self.emb_size//4)//2, kernel_size = 2, stride = 2),
            nn.LeakyReLU(),
            nn.BatchNorm1d((self.emb_size-self.emb_size//4)//2),
            # Rearrange('b e s -> b s e'),
            )
        self.projection_3 =  nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains, in=>B,1,3000 out=>B,64,60
            nn.Conv1d(self.in_channels, self.emb_size//4, kernel_size = 32, stride = 32),
            nn.LeakyReLU(),
            nn.Conv1d(self.emb_size//4, (self.emb_size-self.emb_size//4)//2, kernel_size =2, stride = 2),
            nn.LeakyReLU(),
            nn.BatchNorm1d((self.emb_size-self.emb_size//4)//2),
            # Rearrange('b e s -> b s e'),
            )
        self.projection_4 = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains, in=>B,1,3000 out=>B,64,60
            nn.Conv1d(self.emb_size, self.emb_size, kernel_size = 1, stride = 1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.emb_size),
            Permute(0, 2, 1)
        )
            
        #in=>B,64,60 out=>B,64,61
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.emb_size))
        self.arrange1 = Permute(1, 0, 2)
        #in=>B,61,64 out=>61,B,64
        self.pos = PositionalEncoding(d_model=self.emb_size, dropout=config.network.position_dropout)
        #in=>61,B,64 out=>B,61,64
        self.arrange2 = Permute(1, 0, 2)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(dim = 1)
        b, _, _ = x.shape
        x_1 = self.projection_1(x)
        x_2 = self.projection_2(x)
        x_3 = self.projection_3(x) 
        x = torch.cat([x_1,x_2,x_3],dim = 1)
        x = self.projection_4(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x = self.arrange1(x)
        x = self.pos(x)
        x = self.arrange2(x)
        return x
    
    
class Intra_modal_atten(nn.Module): 
    def __init__(self, config, first:bool =True):
        super().__init__()
        self.d_model=config.network.d_model
        self.nhead = config.network.num_head
        self.drop_out = config.network.atten_dropout
        self.first = first

        if self.first:
            self.window_embed = Window_Embedding(config)
        self.norm = nn.LayerNorm(self.d_model, eps=1e-5)
        self.self_attn = MultiheadAttention(self.d_model, self.nhead, dropout=self.drop_out, batch_first=True)

        self.dropout = nn.Dropout(self.drop_out)
    def forward(self, x: Tensor) -> Tensor:
        if self.first:
            src = self.window_embed(x)
        else:
            src = x

        src2 = self.self_attn(src, src, src)[0]
        out = src + self.dropout(src2)
        out = self.norm(out)
        return out                                

class Cross_modal_atten(nn.Module): 
    def __init__(self, config, first:bool=True) -> None:
        super().__init__()
        self.d_model = config.network.d_model
        self.nhead = config.network.num_head
        self.drop_out = config.network.atten_dropout
        self.first = first

        if self.first:
            self.cls_token = nn.Parameter(torch.randn(1,1, self.d_model))
        self.norm = nn.LayerNorm(self.d_model, eps=1e-5)
        self.cross_attn = MultiheadAttention(self.d_model, self.nhead, dropout=self.drop_out, batch_first=True)

        self.dropout =nn.Dropout(self.drop_out)

    def forward(self, x1: Tensor,x2: Tensor) -> Tensor:
        if len(x1.shape) == 2:
            x = torch.cat([x1.unsqueeze(dim=1), x2.unsqueeze(dim=1)], dim=1)
        else:
            x = torch.cat([x1, x2.unsqueeze(dim=1)], dim=1)
        b,_, _ = x.shape

        if self.first:
            cls_tokens = self.cls_token.expand(b, -1, -1)
            # prepend the cls token to the input
            src = torch.cat([cls_tokens, x], dim=1)  #####
        else:
            src = x
        src2 = self.cross_attn(src, src, src)[0]
        out = src + self.dropout(src2)
        out = self.norm(out)
        return out 
    
class Feed_forward(nn.Module): 
    def __init__(self, config):
        super().__init__()
        self.d_model = config.network.d_model
        self.dim_feedforward = config.network.dim_feedforward
        self.drop_out = config.network.forward_dropout

        self.norm = nn.LayerNorm(self.d_model, eps=1e-5)
        self.linear1 = nn.Linear(self.d_model, self.dim_feedforward)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(self.drop_out)
        self.linear2 = nn.Linear(self.dim_feedforward, self.d_model)
        self.dropout2 = nn.Dropout(self.drop_out)
        
    def forward(self, x: Tensor) -> Tensor:        
        src = x
        src2 = self.linear2(self.dropout1(self.relu(self.linear1(src))))
        out = src + self.dropout2(src2)
        out = self.norm(out)
        return out

# class Epoch_Cross_Transformer_Network(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.eeg_atten = Intra_modal_atten(config,first=True)
#         self.eog_atten = Intra_modal_atten(config,first=True)
#         self.cross_atten = Cross_modal_atten(config,first=True)
#
#         self.eeg_ff = Feed_forward(config)
#         self.eog_ff = Feed_forward(config)
#
#         self.d_model = config.network.d_model
#         self.mlp = nn.Sequential(nn.Flatten(),
#                                  nn.Linear(self.d_model * 2, config.data_loader.num_classes))
#
#     def forward(self, eeg: Tensor, eog: Tensor, finetune = False):
#         self_eeg = self.eeg_atten(eeg)
#         self_eog = self.eog_atten(eog)
#         cross = self.cross_atten(self_eeg[:, 0, :], self_eog[:, 0, :])
#
#         cross_cls = cross[:, 0, :].unsqueeze(dim=1)
#
#         eeg_new = torch.cat([cross_cls, self_eeg[:, 1:, :]], dim=1)
#         eog_new = torch.cat([cross_cls, self_eog[:, 1:, :]], dim=1)
#
#         ff_eeg = self.eeg_ff(eeg_new)
#         ff_eog = self.eog_ff(eog_new)
#
#         cls_out = torch.cat([ff_eeg[:, 0, :], ff_eog[:, 0, :]], dim=1).unsqueeze(dim=1)
#
#         feat_list = [cross_cls, ff_eeg, ff_eog]
#         if finetune == True:
#             out = self.mlp(cls_out)  #########
#             return out, cls_out, feat_list
#         else:
#             return cls_out


class Epoch_Cross_Transformer(nn.Module):
    def __init__(self, config):  # filt_ch = 4
        super().__init__()
        self.eeg_atten = Intra_modal_atten(config,first=True)
        self.eog_atten = Intra_modal_atten(config,first=True)
        self.cross_atten = Cross_modal_atten(config,first=True)

    def forward(self, eeg: Tensor, eog: Tensor):  # ,finetune = False):
        self_eeg = self.eeg_atten(eeg)
        self_eog = self.eog_atten(eog)
        cross = self.cross_atten(self_eeg[:, 0, :], self_eog[:, 0, :])

        cross_cls = cross[:, 0, :].unsqueeze(dim=1)
        feat_list = [self_eeg, self_eog, cross]

        return cross_cls, feat_list


class Seq_Cross_Transformer_Network(nn.Module):
    def __init__(self, config):  # filt_ch = 4
        super().__init__()
        self.d_model = config.network.d_model
        self.nhead = config.network.num_head
        self.drop_out = config.network.dropout
        self.dim_feedforward = config.network.dim_feedforward
        self.window_size = config.network.window_size
        self.num_epochs = config.network.num_epochs  # 10
        self.num_classes = config.data_loader.num_classes  # 5
        self.epoch_len= config.network.epoch_len

        self.epoch_transformers = ModuleList([
            Epoch_Cross_Transformer(config) for _ in range(self.num_epochs)
        ])

        self.seq_atten = Intra_modal_atten(config, first=False)
        self.ff_net = Feed_forward(config)

        self.classification_heads = ModuleList([
            nn.Sequential(nn.Flatten(),
                          nn.Linear(self.d_model, self.num_classes))
            for _ in range(self.num_epochs)
        ])

    def forward(self, x, finetune=False):
        x=x.squeeze(1)
        eeg_all = x[:, 0:1, :]
        eog_all = x[:, 2:3, :]

        epoch_len = self.epoch_len
        epoch_outs = []
        feat_list = []
        for idx, epoch_transformer in enumerate(self.epoch_transformers):
            eeg_epoch = eeg_all[:, :, idx * epoch_len: (idx + 1) * epoch_len]  # [B,3,3840]
            eog_epoch = eog_all[:, :, idx * epoch_len: (idx + 1) * epoch_len]  # [B,1,3840]

            epoch_out, feat = epoch_transformer(eeg_epoch, eog_epoch)
            epoch_outs.append(epoch_out)
            feat_list.append(feat)

        # 拼接所有Epoch的CLS特征，做序列级注意力
        seq = torch.cat(epoch_outs, dim=1)
        seq = self.seq_atten(seq)
        seq = self.ff_net(seq)
        feat_list.append(seq)
        # 每个Epoch的分类结果
        cls_outs_list = [
            head(seq[:, idx, :]) for idx, head in enumerate(self.classification_heads)
        ]

        cls_outs = torch.stack(cls_outs_list, dim=1)
        cls_outs= cls_outs.permute(0, 2, 1)

        if finetune:
            return cls_outs, feat_list, seq
        else:
            return cls_outs
