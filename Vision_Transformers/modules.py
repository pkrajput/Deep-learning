import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch import nn, einsum
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Identity, Parameter, init


class CustomMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim)  # don't change the name
        self.out_proj = nn.Linear(embed_dim, embed_dim)  # don't change the name
        
        self.norm_coeff = self.head_dim ** -0.5

        self.attention_dropout = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):

        # YOUR CODE
        B, N, C = x.shape

        heads = self.num_heads
        qkv = self.in_proj(x).reshape(B, N, 3, heads, C // heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
 
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.norm_coeff
        attn = attn.softmax(dim=-1)
        attn = self.attention_dropout(attn)
        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, N, C)

        
        
        x = self.out_proj(x)
        
        x = self.proj_drop(x)

        #raise NotImplementedError
        return x
from typing import Optional

class StepLRWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, warmup_epochs=3, warmup_lr_init=1e-5,
                 min_lr=1e-5,
                 last_epoch=-1, verbose=False):
        self.step_size = step_size
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_init = warmup_lr_init
        self.min_lr = min_lr
        self.update_steps = step_size
        self.lr = warmup_lr_init
        self.warmup_steps = warmup_epochs

        super().__init__(optimizer, last_epoch, verbose)
        self.init_lr = warmup_lr_init

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if (self.last_epoch == 0):
            y = [self.warmup_lr_init for _ in self.optimizer.param_groups]
            return y
        # YOUR CODE
    
        if self.last_epoch in range(0, self.warmup_epochs + 1):
            ret = [(self.last_epoch * (lr- self.warmup_lr_init ) / self.warmup_epochs) + self.warmup_lr_init for lr in self.base_lrs]
            return ret

        if self.last_epoch== self.warmup_epochs:
            lr = [lr for lr in self.base_lrs]
            return lr
        if (self.last_epoch - self.warmup_epochs) % self.step_size == 0: 
            val = [group['lr']*self.gamma for group in self.optimizer.param_groups]
            return [self.min_lr] if val[0] <= self.min_lr else val 
        if (self.last_epoch - self.warmup_epochs) % self.step_size != 0:
            return [group['lr'] for group in self.optimizer.param_groups]


class TokenizerCCT(nn.Module):
    def __init__(self,
                 kernel_size=3, stride=1, padding=1,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 ):
        super().__init__()
        self.tokenizer_layers = nn.Sequential(
                # YOUR CODE

                nn.Conv2d(n_input_channels, n_output_channels,
                          kernel_size=(kernel_size, kernel_size),
                          stride=(stride, stride),
                          padding=(padding, padding), bias=False),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=pooling_kernel_size,
                             stride=pooling_stride,
                             padding=pooling_padding))

        self.flattener = nn.Flatten(2, 3)  

    def forward(self, x):
        y = self.tokenizer_layers(x)
        y = self.flattener(y)
        y = y.transpose(-2, -1)  
        return y


class SeqPooling(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.attention_pool = nn.Linear(embedding_dim, 1)

    def forward(self, x):

        w = self.attention_pool(x)
        w = F.softmax(w, dim=1)
        
        w = w.transpose(-1, -2)

        
        y = torch.matmul(w , x)

        
        y = y.squeeze(1)

        return y

def create_mlp(embedding_dim, mlp_size, dropout_rate):
    return nn.Sequential(
        # YOUR CODE: Linear + GELU + Dropout + Linear + Dropout
        nn.Linear(embedding_dim,mlp_size ),
        nn.GELU(),
        nn.Dropout(dropout_rate),
        nn.Linear(mlp_size, embedding_dim),
        nn.Dropout(dropout_rate)

    )


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_() 
        output = x.div(keep_prob) * random_tensor
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, mlp_size, dropout=0.1, attention_dropout=0.1,
                 drop_path_rate=0.1):
        super().__init__()
        # YOUR CODE
        self.attention_pre_norm = LayerNorm(embedding_dim)
        self.attention = torch.nn.MultiheadAttention(embedding_dim, num_heads, attention_dropout, dropout)
        self.attention_output_dropout = nn.Dropout(dropout)

        self.mlp_pre_norm = LayerNorm(embedding_dim)
        self.mlp = create_mlp(embedding_dim, mlp_size, dropout)

        if drop_path_rate > 0:
          self.drop_path = DropPath(drop_path_rate)
        else: 
          self.drop_path = nn.Identity()

        #self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):

        y = self.attention_pre_norm(x)
        attention = self.attention(y, y, y)[0]
        attention = self.attention_output_dropout(attention)
        x = x + self.drop_path(attention) 

      
        y = self.mlp_pre_norm(x)
        y = self.mlp(y)
        x = x + self.drop_path(y)  
        return x


class CompactConvTransformer3x1(nn.Module):
    def __init__(self,
                 input_height, input_width,
                 n_tokens,
                 n_input_channels,
                 embedding_dim,
                 num_layers,
                 num_heads=4,
                 num_classes=10,
                 mlp_ratio=2,
                 dropout=0.1,
                 attention_dropout=0.1,
                 stochastic_depth=0.1):
        super().__init__()

        
        pooling_stride = 2
        self.tokenizer = TokenizerCCT(kernel_size=3, stride=1, padding=1,
                                      pooling_kernel_size=3, pooling_stride=pooling_stride, pooling_padding=1,
                                      n_output_channels=embedding_dim)
        n_tokens = input_height // pooling_stride

        
        self.positional_embeddings = torch.nn.Parameter(
            torch.empty((1, n_tokens * n_tokens, embedding_dim)), requires_grad=True)
        torch.nn.init.trunc_normal_(self.positional_embeddings, std=0.2)

        
        mlp_size = int(embedding_dim * mlp_ratio)
        layers_drop_path_rate = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = nn.Sequential(*[
            TransformerEncoder(
                embedding_dim, num_heads, mlp_size,
                dropout=dropout, attention_dropout=attention_dropout,
                drop_path_rate=layers_drop_path_rate[i])
            for i in range(num_layers)])

        
        self.norm = nn.LayerNorm(embedding_dim)

        
        self.pool = SeqPooling(embedding_dim)

        
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):

        patch_embeddings = self.tokenizer(x)


        x = patch_embeddings + self.positional_embeddings


        for block in self.blocks:
            x = block(x)


        x = self.norm(x)

        x = self.pool(x)

        return self.fc(x)
