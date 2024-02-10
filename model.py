from typing import Optional
import faiss
import faiss.contrib.torch_utils  # noqa  << this line makes faiss work with PyTorch
import torch
import torch.nn as nn
from torchvision.transforms.v2 import Lambda
from deep import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Model(nn.Module):
    """
    Model: TabR-S (simple)
    """
    def __init__(self,
                 n_num_features,
                 n_bin_features,
                 n_cat_features,
                 n_classes,
                 d_main = 265, # out dimension of encoder
                 d_multiplier = 2, # linear in block
                 context_dropout = 0.38920071545944357, # dropout after softmax on R
                 dropout = 0.38852797479169876, # dropout in Block
                 normalization = nn.LayerNorm, # normization in Block
                 activation = nn.ReLU, # activation in Block
                 encoder_n_blocks = 0,
                 predictor_n_blocks = 1,
                 segmentation_batch_size = None,
                 ):
        super().__init__()
        """
        Paramètres par défaut du TabR-S
        """
        d_int = int(d_main*d_multiplier)

        def make_block(norm):
            args = [
                nn.Linear(d_main, d_int),
                activation(),
                nn.Dropout(dropout),
                nn.Linear(d_int, d_main),
            ]
            if norm: args.insert(0, normalization(d_main))
            return nn.Sequential(*args)
        
        # Partie Encoder
        self.linear = nn.Linear(
            n_num_features+
            n_bin_features+
            cat_features,
                    d_main)
        self.block_E = nn.ModuleList([make_block(i>0) for i in range(encoder_n_blocks)])

        # Partie Predictor
        out_dim = 1 if (n_classes == 2 or n_classes is None) else n_classes
        self.P = nn.Sequential(
            normalization(d_main),
            activation(),
            nn.Linear(d_main, out_dim)
        )
        self.block_P = nn.ModuleList([make_block(True) for i in range(predictor_n_blocks)])

        # Partie Retrieval Module
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
            nn.Dropout(dropout),
            nn.Linear(d_int, d_main, bias=False)
        )
        self.dropout = nn.Dropout(context_dropout)

        self.segmentation_batch_size = segmentation_batch_size
        self.memory_ki = None

        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.Y, nn.Linear):
            bound = 1 / np.sqrt(2.0)
            nn.init.uniform_(self.Y.weight, -bound, bound)  # type: ignore[code]  # noqa: E501
            nn.init.uniform_(self.Y.bias, -bound, bound)  # type: ignore[code]  # noqa: E501
        else:
            assert isinstance(self.Y[0], nn.Embedding)
            nn.init.uniform_(self.Y[0].weight, -1.0, 1.0)  # type: ignore[code]  # noqa: E501

    def forward(self, x, candidat_x, candidat_y, context_size=96, training = False, memory = False):
        x = self.forward_E(x)
        batch_size, d_main = x.shape
        f = self.normlization
        if f is None: f = lambda x: x
        k = self.K(x if f is None else f(x))

        search_index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d_main) 
        
        with torch.no_grad():
            candidat_size = candidat_y.shape[0]
            if memory and self.memory_ki is not None:
                ki = self.memory_ki
            elif (self.segmentation_batch_size is None) or (candidat_size <= self.segmentation_batch_size):
                ki = self.forward_E(candidat_x)
                ki = self.K(ki if f is None else f(ki))
            else:
                ki = torch.cat(
                    [
                        self.forward_E({ k:f(v[index]) for k,v in candidat_x.items() })
                        for index in make_mini_batch(
                            candidat_size, 
                            self.segmentation_batch_size, 
                            shuffle=False)
                    ]
                )
            if memory: self.memory_ki = ki

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
        x_num, x_bin, x_cat = x.get('num'), x.get('bin'), x.get('cat')
        del x
        x = []
        if x_num is not None: x.append(x_num)
        if x_bin is not None: x.append(x_bin)
        if x_cat is not None: x.append(x_cat)
        x = torch.cat(x, 1)
        x = self.linear(x)
        for block in self.block_E:
            x = x + block(x) 
        return x
    
    def reset_memory(self):
        self.memory_ki = None