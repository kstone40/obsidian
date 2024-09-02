from obsidian.objectives import Objective
import torch
from torch import Tensor

from obsidian import ParamSpace, Target

TORCH_DTYPE = torch.double


class COGS_Objective(Objective):
    
    def __init__(self,
                 X_space: ParamSpace,
                 target: list[Target]) -> None:
        
        super().__init__(mo=True)
        
        self.target = target
        self.X_space = X_space

    def forward(self,
                samples: Tensor,
                X: Tensor) -> Tensor:

        # Ydim = s * b * q * m
        # Xdim = b * q * d
        # if q is 1, it is ommitted
        
        # Vol = 0
        # Metal = 1
        # ML = 2
        # Reagent = 3
        Vol_u = torch.tensor(self.X_space.params[0].decode(X[..., 0].detach().numpy()), dtype=torch.float)
        M_u = torch.tensor(self.X_space.params[1].decode(X[..., 1].detach().numpy()), dtype=torch.float)
        ML_u = torch.tensor(self.X_space.params[2].decode(X[..., 2].detach().numpy()), dtype=torch.float)
        R_u = torch.tensor(self.X_space.params[3].decode(X[..., 3].detach().numpy()), dtype=torch.float)

        # Lig_A = 6
        # Lig_B = 7
        LigA_u = X[..., 6]
        LigB_u = X[..., 7]
        
        ligand_factor = (71630.646*LigA_u + 9213.989*LigB_u)
        cogs_X = 877.801*M_u + ligand_factor*(M_u/ML_u) + 1202.330*R_u + 2.438*Vol_u + 701.999
        
        # AY = t1, ee = t2
        AY = torch.tensor(self.target[0].transform_f(samples[..., 0].detach().numpy(), inverse=True),
                          dtype=torch.float).reshape(samples.shape[0:-1])
        AY[AY < 0] = 0
        ee = torch.tensor(self.target[1].transform_f(samples[..., 1].detach().numpy(), inverse=True),
                          dtype=torch.float).reshape(samples.shape[0:-1])
        
        prod_yield = AY*ee
        neg_cogs = -cogs_X/prod_yield
        
        o = torch.stack([prod_yield, neg_cogs], dim=-1)
        
        return o
