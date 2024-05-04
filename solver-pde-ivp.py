import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Enable LaTeX rendering

plt.rcParams.update({
    'font.size': 12,
    "text.usetex": True,
    "font.family": "serif"
})

# DEFAULT FIG SIZE
FIGSIZE = (10, 8)

torch.manual_seed(42)

# WARNING: EVEN THOUGH THE CODE MAY SUGGEST OTHERWISE, THIS CURRENTLY ONLY WORKS FOR CUBES (0, 1)^3, NO OTHER CUBOIDS.
# For other cuboids would require (minor) adjustments.
####################################################################################################
####################################################################################################
####################################################################################################

class BVP2D:
    def __init__(self, PDE_func, domain_bounds, bcs, g_func=None):
        """
        Initialize a boundary value problem for a 2D second-order LINEAR PDE in a RECTANGULAR domain.
        Args:
            PDE_func (callable): Function that takes inputs xt (2D "positions"), u, u_x, u_t, u_xx, u_tt and returns the PDE residual.
            domain_bounds (dict): The bounds of the RECTANGULAR domain, e.g. {'x': (0, 1), 't': (0, 1)}.
            bcs (dict): Boundary conditions, expected to contain functions for boundaries {'east', 'north', 'west', 'south'}.
            g_func (callable): function satisfying (Dirichlet) boundary conditions, necessary if bar scaling approach being used. Should return a tensor with shape of u (heh).
        """
        self.PDE_func = PDE_func
        self.domain_bounds = domain_bounds
        self.bcs = bcs
        self.g_func = g_func

    def eval_pde(self, xt, u, u_x, u_t, u_xx, u_tt):
        output = self.PDE_func(xt, u, u_x, u_t, u_xx, u_tt)
        if output.dim() == 1:
            return output
        else:
            raise Exception('Residual tensor should be one-dimensional!')
    
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
    
    def u_bar_scaling(self, xt, u_hat):
        return xt[:,0] * (1 - xt[:,0]) * xt[:,1] * (1 - xt[:,1]) * torch.squeeze(u_hat) + self.bvp.g_func(xt)

    def forward(self, xt):
        u_hat = self.stack(xt)
        if self.bar_approach:
            return self.u_bar_scaling(xt, u_hat).view(-1, 1) # Ensure singleton second dimension
        else:
            return u_hat.view(-1, 1) # Ensure singleton second dimension
        
class CustomLoss(nn.Module):
    def __init__(self, bvp, gamma=10, bar_approach=False):
        super().__init__()
        self.bvp = bvp
        self.gamma = gamma
        self.bar_approach = bar_approach
    
    def forward(self, xt, u):
        u.requires_grad_(True)

        # Create gradient vectors for each component
        grad_outputs = torch.ones_like(u)

        # Partial derivatives with respect to each input dimension
        grads = torch.autograd.grad(u, xt, grad_outputs=grad_outputs, create_graph=True)[0]
        u_x, u_t = grads[:, 0].view(-1, 1), grads[:, 1].view(-1, 1)

        # Second derivatives
        u_xx = torch.autograd.grad(u_x, xt, grad_outputs=grad_outputs, create_graph=True)[0][:, 0].view(-1, 1)
        u_tt = torch.autograd.grad(u_t, xt, grad_outputs=grad_outputs, create_graph=True)[0][:, 1].view(-1, 1)

        pde_loss = torch.mean(self.bvp.eval_pde(xt, u, u_x, u_t, u_xx, u_tt) ** 2)

        if self.bar_approach:
            return pde_loss
        else:
            # Boundary conditions loss (Dirichlet)
            bc_loss = 0
            # Assign boundary values using the callable attributes from self.bvp.bcs
            east_mask =  (xt[:, 0] == 1)  # x = 1
            north_mask = (xt[:, 1] == 1)  # t = 1
            west_mask =  (xt[:, 0] == 0)  # x = 0
            south_mask = (xt[:, 1] == 0)  # t = 0

            # Compute boundary errors
            bc_loss += torch.mean((u[east_mask]  - self.bvp.bcs['east'](xt[east_mask, 1])).pow(2))
            bc_loss += torch.mean((u[north_mask] - self.bvp.bcs['north'](xt[north_mask, 0])).pow(2))
            bc_loss += torch.mean((u[west_mask]  - self.bvp.bcs['west'](xt[west_mask, 1])).pow(2))
            bc_loss += torch.mean((u[south_mask] - self.bvp.bcs['south'](xt[south_mask, 0])).pow(2))

            # Return total loss
            return pde_loss + self.gamma * bc_loss
        
def train_model(model, optimiser, bvp, loss_class, xt_train, no_epochs):
    loss_values = [] # list to store loss values

    for epoch in range(no_epochs):
        # Differentiate the optimisation process based on the optimiser type
        if isinstance(optimiser, torch.optim.LBFGS):
            # Define the closure function for LBFGS
            def closure():
                optimiser.zero_grad()

                u_pred = model(xt_train)
                # t_pred.requires_grad_(True)
                loss = loss_class.forward(xt_train, u_pred)
                loss.backward()
                return loss
            # Step through the optimiser
            loss = optimiser.step(closure)
        else: # For first-order optimisers like Adam or SGD
            optimiser.zero_grad()
            u_pred = model(xt_train)
            u_pred.requires_grad_(True)
            loss = loss_class(xt_train, u_pred)

            loss.backward()
            optimiser.step()

        # Store the loss value for this epoch
        loss_values.append(loss.item())

        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss.item():e}")

    return loss_values  # Return the list of loss values

def plot_predictions(model, xt_train_tensor, xt_eval_tensor, eval_nn_at_train=True, exact_sol_func=None, plot_type='surface'):
    # Convert the evaluation tensor to numpy for plotting
    xt_eval_numpy = xt_eval_tensor.detach().numpy()
    
    # Predictions from the neural network
    if eval_nn_at_train:
        u_pred_tensor = model(xt_train_tensor)
        xt_plot = xt_train_tensor
    else:
        u_pred_tensor = model(xt_eval_tensor)
        xt_plot = xt_eval_tensor

    u_pred_numpy = u_pred_tensor.detach().numpy()

    # Reshape for plotting
    num_points_per_dim = int(np.sqrt(xt_plot.shape[0]))
    x = xt_plot[:, 0].view(num_points_per_dim, num_points_per_dim).detach().numpy()
    t = xt_plot[:, 1].view(num_points_per_dim, num_points_per_dim).detach().numpy()
    u_pred_reshaped = u_pred_numpy.reshape(num_points_per_dim, num_points_per_dim)

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(18, 8), subplot_kw={'projection': '3d'} if plot_type == 'surface' else None)
    
    if plot_type == 'surface':
        surf = axs[0].plot_surface(x, t, u_pred_reshaped, cmap='viridis', edgecolor='none')
        axs[0].set_title('Neural Network Predictions')
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('t')
        axs[0].set_zlabel('u')
        fig.colorbar(surf, ax=axs[0], shrink=0.5, aspect=5)
    elif plot_type == 'contour':
        contour = axs[0].contourf(x, t, u_pred_reshaped, cmap='viridis', levels=50)
        axs[0].set_title('Neural Network Predictions')
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('t')
        fig.colorbar(contour, ax=axs[0], shrink=0.5, aspect=5)

    if exact_sol_func is not None:
        u_exact_numpy = exact_sol_func(xt_eval_numpy).reshape(num_points_per_dim, num_points_per_dim)
        if plot_type == 'surface':
            surf = axs[1].plot_surface(x, t, u_exact_numpy, cmap='viridis', edgecolor='none')
            axs[1].set_title('Exact Solution')
            axs[1].set_xlabel('x')
            axs[1].set_ylabel('t')
            axs[1].set_zlabel('u')
            fig.colorbar(surf, ax=axs[1], shrink=0.5, aspect=5)
        elif plot_type == 'contour':
            contour = axs[1].contourf(x, t, u_exact_numpy, cmap='viridis', levels=50)
            axs[1].set_title('Exact Solution')
            axs[1].set_xlabel('x')
            axs[1].set_ylabel('t')
            fig.colorbar(contour, ax=axs[1], shrink=0.5, aspect=5)
    else:
        axs[1].set_title('No exact solution provided')

    plt.tight_layout()
    plt.show()

def plot_loss_vs_epoch(loss_values):
    plt.figure(figsize=FIGSIZE)
    plt.plot(loss_values, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.yscale('log')  # Set the y-axis to logarithmic scale
    plt.legend()
    plt.show()

def plot_pde_residuals(model, bvp, xt_train_tensor):
    # Predictions from the neural network
    u_pred = model(xt_train_tensor)

    # Make sure predictions are detached and require gradients for further computation
    u_pred.requires_grad_(True)

    # Compute derivatives with respect to both dimensions
    grad_outputs = torch.ones_like(u_pred)
    grads = torch.autograd.grad(u_pred, xt_train_tensor, grad_outputs=grad_outputs, create_graph=True)[0]
    u_x, u_t = grads[:, 0].view(-1, 1), grads[:, 1].view(-1, 1)
    u_xx = torch.autograd.grad(u_x, xt_train_tensor, grad_outputs=grad_outputs, create_graph=True)[0][:, 0]
    u_tt = torch.autograd.grad(u_t, xt_train_tensor, grad_outputs=grad_outputs, create_graph=True)[0][:, 1]

    # Evaluate the PDE residuals
    residuals = np.abs(bvp.eval_pde(xt_train_tensor, u_pred, u_x, u_t, u_xx, u_tt).detach().numpy())

    # Reshape for plotting
    num_points_per_dim = int(np.sqrt(xt_train_tensor.shape[0]))
    x = xt_train_tensor[:, 0].view(num_points_per_dim, num_points_per_dim).detach().numpy()
    t = xt_train_tensor[:, 1].view(num_points_per_dim, num_points_per_dim).detach().numpy()
    residuals_reshaped = residuals.reshape(num_points_per_dim, num_points_per_dim)

    # Plotting
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(x, t, residuals_reshaped, cmap='viridis')
    plt.colorbar(contour)
    plt.title('PDE Residuals (abs) Across the Domain')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.show()

# MESH GENERATION
def random_mesh(domain_bounds, num_interior_points, x_bound_points, t_bound_points):
    # Calculate the scaling and offset for the interior points
    x_range = domain_bounds['x'][1] - domain_bounds['x'][0]
    t_range = domain_bounds['t'][1] - domain_bounds['t'][0]
    x_offset = domain_bounds['x'][0]
    t_offset = domain_bounds['t'][0]

    # Interior points
    interior = torch.rand(num_interior_points, 2)
    interior[:, 0] = interior[:, 0] * x_range + x_offset
    interior[:, 1] = interior[:, 1] * t_range + t_offset

    # Boundary points
    x_edges = torch.linspace(domain_bounds['x'][0], domain_bounds['x'][1], steps=x_bound_points)
    t_edges = torch.linspace(domain_bounds['t'][0], domain_bounds['t'][1], steps=t_bound_points)
    boundary = torch.cat([
        torch.stack([x_edges, torch.full_like(x_edges, domain_bounds['t'][0])], dim=1),  # Bottom
        torch.stack([x_edges, torch.full_like(x_edges, domain_bounds['t'][1])], dim=1),  # Top
        torch.stack([torch.full_like(t_edges, domain_bounds['x'][0]), t_edges], dim=1),  # Left
        torch.stack([torch.full_like(t_edges, domain_bounds['x'][1]), t_edges], dim=1)   # Right
    ])

    # Combine interior and boundary points
    xt_train = torch.cat([interior, boundary], dim=0)
    xt_train.requires_grad_(True)
    return xt_train

def uniform_mesh(domain_bounds, x_points, t_points):
    x_points = torch.linspace(domain_bounds['x'][0], domain_bounds['x'][1], steps=x_points)
    t_points = torch.linspace(domain_bounds['t'][0], domain_bounds['t'][1], steps=t_points)

    x_grid, t_grid = torch.meshgrid(x_points, t_points, indexing='ij')  # Create a mesh grid
    xt_train = torch.stack([x_grid.flatten(), t_grid.flatten()], dim=1)  # Flatten and stack to create 2D points

    xt_train.requires_grad_(True)  # Enable gradient tracking

    return xt_train
