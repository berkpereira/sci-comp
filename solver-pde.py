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
            return self.u_bar_scaling(xy, u_hat).view(-1, 1) # Ensure singleton second dimension
        else:
            return u_hat.view(-1, 1) # Ensure singleton second dimension
        
class CustomLoss(nn.Module):
    def __init__(self, bvp, gamma=10, bar_approach=False):
        super().__init__()
        self.bvp = bvp
        self.gamma = gamma
        self.bar_approach = bar_approach
    
    def forward(self, xy, u):
        u.requires_grad_(True)

        # Create gradient vectors for each component
        grad_outputs = torch.ones_like(u)

        # Partial derivatives with respect to each input dimension
        grads = torch.autograd.grad(u, xy, grad_outputs=grad_outputs, create_graph=True)[0]
        u_x, u_y = grads[:, 0].view(-1, 1), grads[:, 1].view(-1, 1)

        # Second derivatives
        u_xx = torch.autograd.grad(u_x, xy, grad_outputs=grad_outputs, create_graph=True)[0][:, 0].view(-1, 1)
        u_yy = torch.autograd.grad(u_y, xy, grad_outputs=grad_outputs, create_graph=True)[0][:, 1].view(-1, 1)

        pde_loss = torch.mean(self.bvp.eval_pde(xy, u, u_x, u_y, u_xx, u_yy) ** 2)

        if self.bar_approach:
            return pde_loss
        else:
            # Boundary conditions loss (Dirichlet)
            bc_loss = 0
            # Assign boundary values using the callable attributes from self.bvp.bcs
            east_mask = (xy[:, 1] == 1)  # y = 1
            north_mask = (xy[:, 0] == 1)  # x = 1
            west_mask = (xy[:, 1] == 0)  # y = 0
            south_mask = (xy[:, 0] == 0)  # x = 0

            # Compute boundary errors
            bc_loss += torch.mean((u[east_mask] - self.bvp.bcs['east'](xy[east_mask, 0])).pow(2))
            bc_loss += torch.mean((u[north_mask] - self.bvp.bcs['north'](xy[north_mask, 1])).pow(2))
            bc_loss += torch.mean((u[west_mask] - self.bvp.bcs['west'](xy[west_mask, 0])).pow(2))
            bc_loss += torch.mean((u[south_mask] - self.bvp.bcs['south'](xy[south_mask, 1])).pow(2))

            # Return total loss
            return pde_loss + self.gamma * bc_loss
        
def train_model(model, optimiser, bvp, loss_class, xy_train, no_epochs):
    loss_values = [] # list to store loss values

    for epoch in range(no_epochs):
        # Differentiate the optimisation process based on the optimiser type
        if isinstance(optimiser, torch.optim.LBFGS):
            # Define the closure function for LBFGS
            def closure():
                optimiser.zero_grad()

                u_pred = model(xy_train)
                # y_pred.requires_grad_(True)
                loss = loss_class.forward(xy_train, u_pred)
                loss.backward()
                return loss
            # Step through the optimiser
            loss = optimiser.step(closure)
        else: # For first-order optimisers like Adam or SGD
            optimiser.zero_grad()
            u_pred = model(xy_train)
            u_pred.requires_grad_(True)
            loss = loss_class(xy_train, u_pred)

            loss.backward()
            optimiser.step()

        # Store the loss value for this epoch
        loss_values.append(loss.item())

        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss.item():e}")

    return loss_values  # Return the list of loss values

####################################################################################################
####################################################################################################
####################################################################################################

BVP_NO = 0
BAR_APPROACH = False

if BVP_NO == 0:
    def laplace_pde(xy, u, u_x, u_y, u_xx, u_yy):
        return u_xx + u_yy

    def boundary_east(x):
        return 0.0
    def boundary_west(x):
        return 0.0
    def boundary_north(y):
        return 0.0
    def boundary_south(y):
        return 0.0 

    # Domain bounds
    domain_bounds = {'x': (0, 1), 'y': (0, 1)}

    no_epochs = 5000
    learning_rate = 0.05


# Boundary conditions dictionary
bcs = {
    'east': boundary_east,
    'north': boundary_north,
    'west': boundary_west,
    'south': boundary_south
}

# BVP instance
bvp = BVP2D(PDE_func=laplace_pde, domain_bounds=domain_bounds, bcs=bcs)

# Neural network parameters
input_features = 2
output_features = 1
hidden_units = 50
depth = 2

# Create the neural network
model = NeuralNetwork2D(bvp, input_features=2, output_features=1, hidden_units=hidden_units, depth=depth, bar_approach=BAR_APPROACH)

# Optimizer
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Loss class instance
loss_class = CustomLoss(bvp)


# (random mesh, FORCED to include BOUNDARY points!)
def random_mesh(domain_bounds, num_interior_points, x_bound_points, y_bound_points):
    # Calculate the scaling and offset for the interior points
    x_range = domain_bounds['x'][1] - domain_bounds['x'][0]
    y_range = domain_bounds['y'][1] - domain_bounds['y'][0]
    x_offset = domain_bounds['x'][0]
    y_offset = domain_bounds['y'][0]

    # Interior points
    interior = torch.rand(num_interior_points, 2)
    interior[:, 0] = interior[:, 0] * x_range + x_offset
    interior[:, 1] = interior[:, 1] * y_range + y_offset

    # Boundary points
    x_edges = torch.linspace(domain_bounds['x'][0], domain_bounds['x'][1], steps=x_bound_points)
    y_edges = torch.linspace(domain_bounds['y'][0], domain_bounds['y'][1], steps=y_bound_points)
    boundary = torch.cat([
        torch.stack([x_edges, torch.full_like(x_edges, domain_bounds['y'][0])], dim=1),  # Bottom
        torch.stack([x_edges, torch.full_like(x_edges, domain_bounds['y'][1])], dim=1),  # Top
        torch.stack([torch.full_like(y_edges, domain_bounds['x'][0]), y_edges], dim=1),  # Left
        torch.stack([torch.full_like(y_edges, domain_bounds['x'][1]), y_edges], dim=1)   # Right
    ])

    # Combine interior and boundary points
    xy_train = torch.cat([interior, boundary], dim=0)
    xy_train.requires_grad_(True)
    return xy_train

def uniform_mesh(domain_bounds, x_points, y_points):
    x_points = torch.linspace(domain_bounds['x'][0], domain_bounds['x'][1], steps=x_points)
    y_points = torch.linspace(domain_bounds['y'][0], domain_bounds['y'][1], steps=y_points)

    x_grid, y_grid = torch.meshgrid(x_points, y_points, indexing='ij')  # Create a mesh grid
    xy_train = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)  # Flatten and stack to create 2D points

    xy_train.requires_grad_(True)  # Enable gradient tracking

    return xy_train

# GENERATE MESH
xy_train = uniform_mesh(domain_bounds, 50, 50)

# Training the model
loss_values = train_model(model, optimiser, bvp, loss_class, xy_train, no_epochs)