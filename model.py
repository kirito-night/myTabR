from typing import Optional
import faiss
import faiss.contrib.torch_utils  # noqa  << this line makes faiss work with PyTorch
import torch
import torch.nn as nn
from torchvision.transforms.v2 import Lambda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Model(nn.Module):
    """
    Model: TabR-S (simple)
    """
    def __init__(self,
                 n_num_features: int,
                 n_bin_features: int,
                 cat_cardinalities: list[int],
                 n_classes: Optional[int],
                 d_main: int, # out dimension of encoder
                 d_multiplier: float, # linear in block
                 context_dropout: float, # dropout after softmax on R
                 dropout0: float, # dropout in Block
                 normalization: nn.LayerNorm, # normization in Block
                 activation: nn.ReLU, # activation in Block
                 encoder_n_blocks: int = 0,
                 predictor_n_blocks: int = 1,
                 ):
        super().__init__()
        d_int = int(d_main*d_multiplier)

        def make_block(norm):
            args = [
                nn.Linear(d_main, d_int),
                activation(),
                nn.Dropout(dropout0),
                nn.Linear(d_int, d_main)
            ]
            if norm: args.insert(0, normalization(d_main))
            return nn.Sequential(*args)
        
        # E
        self.linear = nn.Linear(
            n_num_features+
            n_bin_features+
            cat_cardinalities,
                    d_main)
        self.block_E = nn.ModuleList([make_block(i>0) for i in range(encoder_n_blocks)])

        # P
        out_dim = 1 if (n_classes == 2 or n_classes is None) else n_classes
        self.P = nn.Sequential(
            normalization(d_main),
            activation(),
            nn.Linear(d_main, out_dim)
        )
        self.block_P = nn.ModuleList([make_block(True) for i in range(predictor_n_blocks)])

        # R
        self.normlization = nn.LayerNorm(d_main) if encoder_n_blocks > 0 else None
        self.K = nn.Linear(d_main, d_main)
        self.Y = (
            nn.Linear(1, d_main)
            if n_classes is None 
            else nn.Sequential(
                nn.Embedding(n_classes, d_main),
                Lambda(lambda x: x.squeeze(-2))
            )
        )
        self.T = nn.Sequential(
            nn.Linear(d_main, d_int),
            activation(),
            nn.Dropout(dropout0),
            nn.Linear(d_int, d_main, bias=False)
        )
        self.dropout = nn.Dropout(context_dropout)
        self.search_index = (
            faiss.GpuIndexFlatL2(faiss.StandardGpuResources(),d_main)
        )

    def forward(self, x, candidat_x, candidat_y, context_size=96, training = False):
        x = self.forward_E(x)
        batch_size, d_main = x.shape
        f = self.normlization
        k = self.K(x if f is None else f(x))
        with torch.no_grad():
            ki = self.forward_E(candidat_x)
            ki = self.K(ki if f is None else f(ki))
            self.search_index.reset()
            self.search_index.add(ki)
            D, I = self.search_index.search(k, context_size + (1 if training else 0))
            if training: I = I.gather(-1 , D.argsort()[:, 1:])
            else: ki = ki[I]
        
        if training:
            ki = self.forward_E(
                {
                    key:values[I.flatten()]
                    for key, values in candidat_x.items()
                }
            ).reshape(batch_size, context_size, d_main)
            ki = self.K(ki if f is None else f(ki))

        S = - (
            -k.square().sum(-1, keepdim=True)
            + (2 * (k[..., None, :] @ ki.transpose(-1, -2))).squeeze(-2)
            -ki.square().sum(-1)
        )
        weights = self.dropout(torch.softmax(S, dim=-1))
        encode_y = self.Y(candidat_y[I])
        V = encode_y +  self.T(k[:, None] - ki)
        V = (weights[:, None] @ V).squeeze(1)

        x = x + V
        for block in self.block_P:
            x = x + block(x)
        return self.P(x)
            
    def forward_E(self, x):
        """
        x: Dict[Tensor]
        """
        num, bin, cat = x.get('num'), x.get('bin'), x.get('cat')
        x = []
        if num is not None: x.append(num)
        if bin is not None: x.append(bin)
        if cat is not None: x.append(cat)
        x = torch.cat(x, 1)
        x = self.linear(x)
        for block in self.block_E:
            x = x + block(x) 
        return x
    
