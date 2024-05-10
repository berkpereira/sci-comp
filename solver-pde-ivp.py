import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

# Enable LaTeX rendering

plt.rcParams.update({
    'font.size': 11,
    "text.usetex": True,
    "font.family": "serif",
    "axes.grid": True,
    'grid.alpha': 0.5
})

plot_path = '/Users/gabrielpereira/OneDrive - Nexus365/ox-mmsc-cloud/computing-report/report/plots/ivp-2d/'

# DEFAULT FIG SIZE
FIGSIZE = (6, 2)

torch.manual_seed(42)

# WARNING: EVEN THOUGH THE CODE MAY SUGGEST OTHERWISE, THIS CURRENTLY ONLY WORKS FOR SPATIAL DOMAINS x in (0, 1).
# For other dimensions would require (minor) adjustments.
# Also TIME DOMAIN MUST BEGIN at t = 0.
####################################################################################################
####################################################################################################
####################################################################################################

class IVP2D:
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
    def __init__(self, ivp, input_features=2, output_features=1, hidden_units=50, depth=1, bar_approach=False):
        super().__init__()
        self.ivp = ivp
        self.bar_approach = bar_approach
        

        if depth == 0: # EDGE CASE, JUST LINEAR
            layers = [nn.Linear(in_features=input_features, out_features=output_features)]
        else:
            layers = [nn.Linear(in_features=input_features, out_features=hidden_units), nn.Sigmoid()]
            
            for _ in range(depth - 1):
                layers.append(nn.Linear(in_features=hidden_units, out_features=hidden_units))
                layers.append(nn.Sigmoid())

            # Add the final layer
            layers.append(nn.Linear(in_features=hidden_units, out_features=output_features))
        
        # Create the sequential model
        self.stack = nn.Sequential(*layers)
    
    def u_bar_scaling(self, xt, u_hat):
        return xt[:,0] * (1 - xt[:,0]) * xt[:,1] * torch.squeeze(u_hat) + self.ivp.g_func(xt)

    def forward(self, xt):
        u_hat = self.stack(xt)
        if self.bar_approach:
            return self.u_bar_scaling(xt, u_hat).view(-1, 1) # Ensure singleton second dimension
        else:
            return u_hat.view(-1, 1) # Ensure singleton second dimension
        
class CustomLoss(nn.Module):
    def __init__(self, ivp, gamma=10, bar_approach=False):
        super().__init__()
        self.ivp = ivp
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

        pde_loss = torch.mean(self.ivp.eval_pde(xt, u, u_x, u_t, u_xx, u_tt) ** 2)

        if self.bar_approach:
            return pde_loss
        else:
            # Boundary conditions loss (Dirichlet)
            bc_loss = 0
            # Assign boundary values using the callable attributes from self.ivp.bcs
            east_mask =  (xt[:, 0] == 1)  # x = 1
            west_mask =  (xt[:, 0] == 0)  # x = 0
            south_mask = (xt[:, 1] == 0)  # t = 0

            # Number of points on the boundary
            num_boundary_points = east_mask.sum() + west_mask.sum() + south_mask.sum()

            # Compute boundary errors
            bc_loss += torch.sum((u[east_mask]  - self.ivp.bcs['east'](xt[east_mask, 1])).pow(2))
            bc_loss += torch.sum((u[west_mask]  - self.ivp.bcs['west'](xt[west_mask, 1])).pow(2))
            bc_loss += torch.sum((u[south_mask] - self.ivp.bcs['south'](xt[south_mask, 0])).pow(2))

            bc_loss = bc_loss / num_boundary_points # take mean
            
            # Return total loss
            return pde_loss + self.gamma * bc_loss
        
def train_model(model, optimiser, ivp, loss_class, xt_train, no_epochs):
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

def plot_predictions(model, xt_train_tensor, xt_eval_tensor,  cbar_ticks=None, eval_nn_at_train=True, exact_sol_func=None, plot_type='surface', savefig=False, plot_path=None):
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
    fig, axs = plt.subplots(1, 2, figsize=FIGSIZE, subplot_kw={'projection': '3d'} if plot_type == 'surface' else None)
    
    # Colour bar specs
    colourbar_dict = {'aspect': 10, 'shrink': 0.5}

    if plot_type == 'surface':
        surf = axs[0].plot_surface(x, t, u_pred_reshaped, cmap='cividis', edgecolor='none')
        # axs[0].set_title('Neural Network Predictions')
        axs[0].set_xlabel('\(x\)')
        axs[0].set_ylabel('\(t\)')
        axs[0].set_zlabel('\(u\)')
        cbar0 = fig.colorbar(surf, ax=axs[0], shrink=colourbar_dict['shrink'], aspect=colourbar_dict['aspect'])
        cbar0.set_label('\(u\)')
    elif plot_type == 'contour':
        contour = axs[0].contourf(x, t, u_pred_reshaped, cmap='cividis', levels=50)
        # axs[0].set_title('Neural Network Predictions')
        axs[0].set_xlabel('\(x\)')
        axs[0].set_ylabel('\(t\)')
        cbar0 = fig.colorbar(contour, ax=axs[0], shrink=colourbar_dict['shrink'], aspect=colourbar_dict['aspect'])
        cbar0.set_label('\(u\)')
    
    # Set colour bar ticks
    if cbar_ticks is not None:
        cbar0.set_ticks(cbar_ticks)

    if exact_sol_func is not None:
        u_exact_numpy = exact_sol_func(xt_eval_numpy).reshape(num_points_per_dim, num_points_per_dim)
        if plot_type == 'surface':
            surf = axs[1].plot_surface(x, t, u_exact_numpy, cmap='cividis', edgecolor='none')
            # axs[1].set_title('Exact Solution')
            axs[1].set_xlabel('\(x\)')
            axs[1].set_ylabel('\(t\)')
            axs[1].set_zlabel('\(u\)')
            cbar1 = fig.colorbar(surf, ax=axs[1], shrink=colourbar_dict['shrink'], aspect=colourbar_dict['aspect'])
            cbar1.set_label('\(u\)')
        elif plot_type == 'contour':
            contour = axs[1].contourf(x, t, u_exact_numpy, cmap='cividis', levels=50)
            # axs[1].set_title('Exact Solution')
            axs[1].set_xlabel('\(x\)')
            axs[1].set_ylabel('\(t\)')
            cbar1 = fig.colorbar(contour, ax=axs[1], shrink=colourbar_dict['shrink'], aspect=colourbar_dict['aspect'])
            cbar1.set_label('\(u\)')

        # Set colour bar ticks
        if cbar_ticks is not None:
            cbar1.set_ticks(cbar_ticks)

    plt.tight_layout()

    if savefig:
        if plot_path is None:
            raise Exception('Must provide parent directory for figure file!')
        if plot_type == 'contour':
            plot_path = plot_path + 'sols-contour.pdf'
        elif plot_type == 'surface':
            plot_path = plot_path + 'sols-surface.pdf'
        else:
            raise Exception('Unrecognised plot type!')
        
        plt.savefig(plot_path)

    plt.show()

def plot_loss_vs_epoch(loss_values, savefig=False, plot_path=None):
    plt.figure(figsize=FIGSIZE)
    plt.plot(loss_values, color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.title('Loss vs. Epoch')
    plt.yscale('log')  # Set the y-axis to logarithmic scale
    plt.legend()
    plt.tight_layout()

    if savefig:
        if plot_path is None:
            raise Exception('Must provide parent directory for figure file!')
        plot_path = plot_path + 'loss-epoch.pdf'
        plt.savefig(plot_path)

    plt.show()

def plot_pde_residuals(model, ivp, xt_train_tensor, savefig=False, plot_path=None):
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
    residuals = np.abs(ivp.eval_pde(xt_train_tensor, u_pred, u_x, u_t, u_xx, u_tt).detach().numpy())

    # Reshape for plotting
    num_points_per_dim = int(np.sqrt(xt_train_tensor.shape[0]))
    x = xt_train_tensor[:, 0].view(num_points_per_dim, num_points_per_dim).detach().numpy()
    t = xt_train_tensor[:, 1].view(num_points_per_dim, num_points_per_dim).detach().numpy()
    residuals_reshaped = residuals.reshape(num_points_per_dim, num_points_per_dim)

    # Plotting
    plt.figure(figsize=FIGSIZE)
    contour = plt.contourf(x, t, residuals_reshaped, cmap='cividis')
    cbar = plt.colorbar(contour)
    cbar.set_label('Residual (abs.\ value)')
    # plt.title('PDE Residuals (abs) Across the Domain')
    plt.xlabel('\(x\)')
    plt.ylabel('\(t\)')
    plt.tight_layout()

    if savefig:
        if plot_path is None:
            raise Exception('Must provide parent directory for figure file!')
        plot_path_modified = plot_path + 'residuals.pdf'
        plt.savefig(plot_path_modified)

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

####################################################################################################
####################################################################################################
####################################################################################################

IVP_NO = 1
BAR_APPROACH = True
OPTIMISER_NAME = 'adam' # adam, lbfgs
NO_POINTS_DIR = 10
MESH_TYPE = 'uniform' # uniform, random

hidden_units = 10
depth = 1

########################################################################################################################
########################################################################################################################

SAVE_FIGURE = False

########################################################################################################################
########################################################################################################################

if IVP_NO == 0:
    # Heat equation, TRIVIAL
    alpha = 1
    def PDE_func(xt, u, u_x, u_t, u_xx, u_tt):
        return torch.squeeze(u_t) - alpha * torch.squeeze(u_xx)

    def boundary_east(x):
        return 0.0
    def boundary_west(x):
        return 0.0
    def boundary_south(t):
        return 0.0 

    # Domain bounds
    domain_bounds = {'x': (0, 1), 't': (0, 3)}

    # Function satisfying boundary conditions, for BAR APPROACH
    def g_func(xt):
        return torch.zeros(xt.size(0))

    def exact_sol(xt):
        # Since the boundary conditions and the PDE suggest a trivial solution:
        return torch.zeros(xt.shape[0], 1)

    no_epochs = 1500
    learning_rate = 0.02

    gamma=10
elif IVP_NO == 1:
    # Heat equation
    alpha = 1
    def PDE_func(xt, u, u_x, u_t, u_xx, u_tt):
        return torch.squeeze(u_t) - alpha * torch.squeeze(u_xx)

    def boundary_east(t):
        return 0.0
    def boundary_west(t):
        return 0.0
    def boundary_south(x):
        return torch.sin(np.pi * x)

    # Domain bounds
    domain_bounds = {'x': (0, 1), 't': (0, 0.1)}

    # Function satisfying boundary conditions, for BAR APPROACH
    def g_func(xt):
        return torch.sin(np.pi * torch.squeeze(xt[:,0]))

    def exact_sol(xt):
        return np.exp(- np.pi**2 * xt[:,1]) * np.sin(np.pi * xt[:,0])

    no_epochs = 10000
    learning_rate = 0.05

    cbar_ticks = np.linspace(0, 1, 5)
    gamma=10
elif IVP_NO == 2:
        # Heat equation
        # LBFGS, 50 wide, 1 deep, with learning rate 0.02 works WAY better than Adam in this one (about 300--400 epochs)
    alpha = 1
    def PDE_func(xt, u, u_x, u_t, u_xx, u_tt):
        return torch.squeeze(u_t) - alpha * torch.squeeze(u_xx)

    def boundary_east(t):
        return 0.0
    def boundary_west(t):
        return 0.0
    def boundary_south(x):
        return torch.sin(np.pi * x) + 0.5 * torch.sin(3 * np.pi * x)

    # Domain bounds
    domain_bounds = {'x': (0, 1), 't': (0, 0.2)}

    # Function satisfying boundary conditions, for BAR APPROACH
    def g_func(xt):
        return torch.sin(np.pi * xt[:,0]) + 0.5 * torch.sin(3 * np.pi * xt[:,0])

    def exact_sol(xt):
        return np.exp(- np.pi**2 * xt[:,1]) * np.sin(np.pi * xt[:,0]) + 0.5 * np.exp(- 9 * np.pi**2 * xt[:,1]) * np.sin(3 * np.pi * xt[:,0])

    no_epochs = 10000
    learning_rate = 0.004

    gamma=5
elif IVP_NO == 3:
    # Heat equation, sawtooth wave
    alpha = 1
    N = 3 # must be positive integer

    # works decently well with alpha = 1, N = 3 (weak...), Adam, lr = 0.01, 30,000 epochs, 50 points in x, 20 in t (from 0 to 0.1),
    # 50 wide, 1 deep

    def PDE_func(xt, u, u_x, u_t, u_xx, u_tt):
        return torch.squeeze(u_t) - alpha * torch.squeeze(u_xx)

    def boundary_east(t):
        return 0.0
    def boundary_west(t):
        return 0.0
    def boundary_south(x):
        n = torch.arange(1, N+1, dtype=torch.float32)  # Create a tensor for n values from 1 to N
        n = n.view(1, -1)  # Reshape for broadcasting (1, N)
        sin_terms = torch.sin(n * np.pi * x.view(-1, 1))  # Shape (batch_size, N)
        output = torch.sum((1.0 / n) * sin_terms, dim=1)
        return output

    # Domain bounds
    domain_bounds = {'x': (0, 1), 't': (0, 0.1)}

    # Function satisfying boundary conditions, for BAR APPROACH
    def g_func(xt):
        return boundary_south(xt[:,0])

    def exact_sol(xt):
        x = xt[:, 0]
        t = xt[:, 1]
        
        n = np.arange(1, N + 1)  # Array of n values from 1 to N
        n = n.reshape(1, -1)  # Reshape for broadcasting (1, N)
        
        # Calculate the terms of the series
        sin_terms = np.sin(n * np.pi * x.reshape(-1, 1))  # Shape (num_samples, N)
        decay_factors = np.exp(-n**2 * np.pi**2 * alpha * t.reshape(-1, 1))  # Shape (num_samples, N)
        
        # Weighted sum of terms (weights could be 1/n or any specific sequence depending on the problem)
        result = np.sum((1.0 / n) * decay_factors * sin_terms, axis=1)  # Sum along the second dimension
        
        return result

    no_epochs = 30000
    learning_rate = 0.01

    gamma=5

# INFORMATIVE FILE NAME FOR SAVING
plot_path = plot_path + f'problem{str(IVP_NO)}/depth{depth}-width{hidden_units}-bar{BAR_APPROACH}-mesh{MESH_TYPE}-points{NO_POINTS_DIR}-optimiser{OPTIMISER_NAME}-epochs{no_epochs}-lr{learning_rate}-gamma{gamma}-'

# Ensure the directory exists
os.makedirs(os.path.dirname(plot_path), exist_ok=True)

####################################################################################################
####################################################################################################
####################################################################################################

# Boundary conditions dictionary
bcs = {
    'east':  boundary_east,
    'west':  boundary_west,
    'south': boundary_south
}

# IVP instance
ivp = IVP2D(PDE_func=PDE_func, domain_bounds=domain_bounds, bcs=bcs, g_func=g_func)

# Create the neural network
model = NeuralNetwork2D(ivp, hidden_units=hidden_units, depth=depth, bar_approach=BAR_APPROACH)


print('------------------------------------------------------------')
print(f'MODEL HAS {sum(p.numel() for p in model.parameters() if p.requires_grad)} TRAINABLE PARAMETERS')
print('------------------------------------------------------------')

# Optimizer
if OPTIMISER_NAME == 'adam':
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
elif OPTIMISER_NAME == 'lbfgs':
    optimiser = torch.optim.LBFGS(model.parameters(), lr=learning_rate)

# Loss class instance
loss_class = CustomLoss(ivp, gamma=gamma, bar_approach=BAR_APPROACH)

# GENERATE MESHES
if MESH_TYPE == 'uniform':
    xt_train = uniform_mesh(domain_bounds, NO_POINTS_DIR, NO_POINTS_DIR)
elif MESH_TYPE == 'random':
    xy_train = random_mesh(domain_bounds, NO_POINTS_DIR, 50, 50)

xt_eval  = uniform_mesh(domain_bounds, 50, 50)

# Training the model
loss_values = train_model(model, optimiser, ivp, loss_class, xt_train, no_epochs)

print('------------------------------------------------------------')
print(f'FINAL LOSS ACHIEVED: {loss_values[-1]:.2e}')
print('------------------------------------------------------------')

# PLOTTING
plot_predictions(model, xt_train, xt_eval, cbar_ticks=cbar_ticks, eval_nn_at_train=False, exact_sol_func=exact_sol, plot_type='surface', savefig=SAVE_FIGURE, plot_path=plot_path)
plot_predictions(model, xt_train, xt_eval, cbar_ticks=cbar_ticks, eval_nn_at_train=False, exact_sol_func=exact_sol, plot_type='contour', savefig=SAVE_FIGURE, plot_path=plot_path)
plot_loss_vs_epoch(loss_values, savefig=SAVE_FIGURE, plot_path=plot_path)
plot_pde_residuals(model, ivp, xt_train, savefig=SAVE_FIGURE, plot_path=plot_path)