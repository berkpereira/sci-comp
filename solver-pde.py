import numpy as np
#from plotly.offline import iplot
#import plotly.graph_objs as go
import torch
from torch import nn
import matplotlib.pyplot as plt

# Enable LaTeX rendering

plt.rcParams.update({
    'font.size': 12,
    "text.usetex": True,
    "font.family": "serif"
})

# DEFAULT FIG SIZE
FIGSIZE = (10, 8)

torch.manual_seed(42)

####################################################################################################
####################################################################################################
####################################################################################################

class BVP2D:
    def __init__(self, PDE_func, domain_bounds, bcs, g_func=None):
        """
        Initialize a boundary value problem for a 2D second-order LINEAR PDE in a RECTANGULAR domain.
        Args:
            PDE_func (callable): Function that takes inputs xy (2D positions), u, u_x, u_y, u_xx, u_yy and returns the PDE residual.
            domain_bounds (dict): The bounds of the RECTANGULAR domain, e.g. {'x': (0, 1), 'y': (0, 1)}.
            bcs (dict): Boundary conditions, expected to contain functions for boundaries {'east', 'north', 'west', 'south'}.
            g_func (callable): function satisfying (Dirichlet) boundary conditions, necessary if bar scaling approach being used. Should return a tensor with shape of u (heh).
        """
        self.PDE_func = PDE_func
        self.domain_bounds = domain_bounds
        self.bcs = bcs

    def eval_pde(self, xy, u, u_x, u_y, u_xx, u_yy):
        return self.PDE_func(xy, u, u_x, u_y, u_xx, u_yy)
    
class NeuralNetwork2D(nn.Module):
    def __init__(self, bvp, input_features=2, output_features=1, hidden_units=50, depth=1, bar_approach=False):
        super().__init__()
        self.bvp = bvp
        self.bar_approach = bar_approach
        
        layers = [nn.Linear(in_features=input_features, out_features=hidden_units), nn.Sigmoid()]
        
        # Add hidden layers based on the depth parameter
        for _ in range(depth - 1):
            layers.append(nn.Linear(in_features=hidden_units, out_features=hidden_units))
            layers.append(nn.Sigmoid())

        # Add the final layer
        layers.append(nn.Linear(in_features=hidden_units, out_features=output_features))
        
        # Create the sequential model
        self.stack = nn.Sequential(*layers)
    
    def u_bar_scaling(self, xy, u_hat):
        return xy[:,0] * (1 - xy[:,0]) * xy[:,1] * (1 - xy[:,1]) * u_hat + self.bvp.g_func

    def forward(self, xy):
        u_hat = self.stack(xy)
        if self.bar_approach:
            return self.u_bar_scaling(xy, u_hat)
        else:
            return u_hat