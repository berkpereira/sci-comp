import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

# Enable LaTeX rendering

plt.rcParams.update({
    'font.size': 11,
    "text.usetex": True,
    "font.family": "serif"
})

plot_file_base = '~/OneDrive - Nexus365/ox-mmsc-cloud/computing-report/report/plots/bvp-3d-'

# DEFAULT FIG SIZE
FIGSIZE = (6, 4)

torch.manual_seed(42)

# WARNING: EVEN THOUGH THE CODE MAY SUGGEST OTHERWISE, THIS CURRENTLY ONLY WORKS FOR CUBES (0, 1)^3, NO OTHER CUBOIDS.
# For other cuboids would require (minor) adjustments.
####################################################################################################
####################################################################################################
####################################################################################################

class BVP3D:
    def __init__(self, PDE_func, domain_bounds, bcs, g_func=None):
        """
        Initialize a boundary value problem for a 3D second-order LINEAR PDE in a RECTANGULAR domain.
        Args:
            PDE_func (callable): Function that takes inputs xyz (3D positions), u, u_x, u_y, u_z, u_xx, u_yy, u_zz and returns the PDE residual (1D tensor).
            domain_bounds (dict): The bounds of the RECTANGULAR domain, e.g. {'x': (0, 1), 'y': (0, 1), 'z': (0, 1)}.
            bcs (dict): Boundary conditions, expected to contain functions for boundaries {'east', 'north', 'west', 'south', 'bottom', 'top'}.
            g_func (callable): function satisfying (Dirichlet) boundary conditions, necessary if bar scaling approach being used. Should return a tensor with shape of u (heh).
        """
        self.PDE_func = PDE_func
        self.domain_bounds = domain_bounds
        self.bcs = bcs
        self.g_func = g_func

    def eval_pde(self, xyz, u, u_x, u_y, u_z, u_xx, u_yy, u_zz):
        output = self.PDE_func(xyz, u, u_x, u_y, u_z, u_xx, u_yy, u_zz)
        if output.dim() == 1:
            return output
        else:
            raise Exception('Residual tensor should be one-dimensional!')
    
class NeuralNetwork3D(nn.Module):
    def __init__(self, bvp, input_features=3, output_features=1, hidden_units=50, depth=1, bar_approach=False):
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
    
    def u_bar_scaling(self, xyz, u_hat):
        return xyz[:,0] * (1 - xyz[:,0]) * xyz[:,1] * (1 - xyz[:,1]) * xyz[:,2] * (1 - xyz[:,2]) * torch.squeeze(u_hat) + self.bvp.g_func(xyz)

    def forward(self, xyz):
        u_hat = self.stack(xyz)
        if self.bar_approach:
            return self.u_bar_scaling(xyz, u_hat).view(-1, 1) # Ensure singleton second dimension
        else:
            return u_hat.view(-1, 1) # Ensure singleton second dimension
        
class CustomLoss(nn.Module):
    def __init__(self, bvp, gamma=10, bar_approach=False):
        super().__init__()
        self.bvp = bvp
        self.gamma = gamma
        self.bar_approach = bar_approach
    
    def forward(self, xyz, u):
        u.requires_grad_(True)

        # Create gradient vectors for each component
        grad_outputs = torch.ones_like(u)

        # Partial derivatives with respect to each input dimension
        grads = torch.autograd.grad(u, xyz, grad_outputs=grad_outputs, create_graph=True)[0]
        u_x, u_y, u_z = grads[:, 0].view(-1, 1), grads[:, 1].view(-1, 1), grads[:, 2].view(-1, 1)

        # Second derivatives
        u_xx = torch.autograd.grad(u_x, xyz, grad_outputs=grad_outputs, create_graph=True)[0][:, 0].view(-1, 1)
        u_yy = torch.autograd.grad(u_y, xyz, grad_outputs=grad_outputs, create_graph=True)[0][:, 1].view(-1, 1)
        u_zz = torch.autograd.grad(u_z, xyz, grad_outputs=grad_outputs, create_graph=True)[0][:, 2].view(-1, 1)

        pde_loss = torch.mean(self.bvp.eval_pde(xyz, u, u_x, u_y, u_z, u_xx, u_yy, u_zz) ** 2)

        if self.bar_approach:
            return pde_loss
        else:
            # Boundary conditions loss (Dirichlet)
            bc_loss = 0
            # Assign boundary values using the callable attributes from self.bvp.bcs
            east_mask   = (xyz[:, 0] == 1)  # x = 1
            north_mask  = (xyz[:, 1] == 1)  # y = 1
            west_mask   = (xyz[:, 0] == 0)  # x = 0
            south_mask  = (xyz[:, 1] == 0)  # y = 0
            bottom_mask = (xyz[:, 2] == 0)  # z = 0
            top_mask    = (xyz[:, 2] == 1)  # z = 1
            
            # Compute boundary errors.
            # NOTE HOW BOUNDARY CONDITION FUNCTIONS NOW SPECIFIED WITH SEPARATE ARGUMENTS FOR EACH COORDINATE (COMPARE/CONTRAST WITH 2D BVP SOLVER CODE)
            bc_loss += torch.mean((u[east_mask]   - self.bvp.bcs['east'](xyz[east_mask, 1], xyz[east_mask, 2])).pow(2))       # func of (y, z)
            bc_loss += torch.mean((u[north_mask]  - self.bvp.bcs['north'](xyz[north_mask, 0], xyz[north_mask, 2])).pow(2))    # func of (x, z)
            bc_loss += torch.mean((u[west_mask]   - self.bvp.bcs['west'](xyz[west_mask, 1], xyz[west_mask, 2])).pow(2))       # func of (y, z)
            bc_loss += torch.mean((u[south_mask]  - self.bvp.bcs['south'](xyz[south_mask, 0], xyz[south_mask, 2])).pow(2))    # func of (x, z)
            bc_loss += torch.mean((u[bottom_mask] - self.bvp.bcs['bottom'](xyz[bottom_mask, 0], xyz[bottom_mask, 1])).pow(2)) # func of (x, y)
            bc_loss += torch.mean((u[top_mask]    - self.bvp.bcs['top'](xyz[top_mask, 0], xyz[top_mask, 1])).pow(2))          # func of (x, y)

            # Return total loss
            return pde_loss + self.gamma * bc_loss
        
def train_model(model, optimiser, bvp, loss_class, xyz_train, no_epochs):
    loss_values = [] # list to store loss values

    for epoch in range(no_epochs):
        # Differentiate the optimisation process based on the optimiser type
        if isinstance(optimiser, torch.optim.LBFGS):
            # Define the closure function for LBFGS
            def closure():
                optimiser.zero_grad()

                u_pred = model(xyz_train)
                # y_pred.requires_grad_(True)
                loss = loss_class.forward(xyz_train, u_pred)
                loss.backward()
                return loss
            # Step through the optimiser
            loss = optimiser.step(closure)
        else: # For first-order optimisers like Adam or SGD
            optimiser.zero_grad()
            u_pred = model(xyz_train)
            u_pred.requires_grad_(True)
            loss = loss_class(xyz_train, u_pred)

            loss.backward()
            optimiser.step()

        # Store the loss value for this epoch
        loss_values.append(loss.item())

        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss.item():e}")

    return loss_values  # Return the list of loss values

# PLOTTING FUNCTIONS
def plot_isosurface_mlab(model, xyz_eval, level, exact_sol_func=None):
    xyz_eval_numpy = xyz_eval.detach().numpy()
    grid_dim = int(np.cbrt(xyz_eval_numpy.shape[0]))  # Assuming a cubic grid
    x_eval, y_eval, z_eval = xyz_eval_numpy[:, 0].reshape(grid_dim, grid_dim, grid_dim), xyz_eval_numpy[:, 1].reshape(grid_dim, grid_dim, grid_dim), xyz_eval_numpy[:, 2].reshape(grid_dim, grid_dim, grid_dim)

    mlab.figure(bgcolor=(1, 1, 1))
    u_pred_tensor = model(xyz_eval) # tensor input
    u_pred_numpy = u_pred_tensor.detach().numpy().reshape(grid_dim, grid_dim, grid_dim)

    src = mlab.contour3d(x_eval, y_eval, z_eval, u_pred_numpy, contours=[level], opacity=0.5)
    
    mlab.title('Neural Network Solution', color=(0, 0, 0))
    mlab.axes(src, color=(0, 0, 0), xlabel='x', ylabel='y', zlabel='z')
    mlab.colorbar(object=src, title='Value', label_fmt='%.2f')
    
    # Adjust colorbar text color to black
    color_bar = mlab.colorbar(object=src, title='Value', label_fmt='%.2f')
    color_bar.label_text_property.color = (0, 0, 0)
    color_bar.title_text_property.color = (0, 0, 0)

    mlab.show()

    if exact_sol_func is not None:
        u_exact_numpy = exact_sol_func(xyz_eval_numpy).reshape(grid_dim, grid_dim, grid_dim)
        mlab.figure(bgcolor=(1, 1, 1))
        src = mlab.contour3d(x_eval, y_eval, z_eval, u_exact_numpy, contours=[level], opacity=0.5)
        mlab.title('Exact Solution', color=(0, 0, 0))
        mlab.axes(src, color=(0, 0, 0), xlabel='x', ylabel='y', zlabel='z')
        mlab.colorbar(object=src, title='Value', label_fmt='%.2f')
        
        # Adjust colorbar text color to black
        color_bar = mlab.colorbar(object=src, title='Value', label_fmt='%.2f')
        color_bar.label_text_property.color = (0, 0, 0)
        color_bar.title_text_property.color = (0, 0, 0)
        mlab.show()

def plot_volume_rendering_mlab(model, xyz_eval, eval_nn_at_train=False, exact_sol_func=None):
    xyz_eval_numpy = xyz_eval.detach().numpy()
    grid_dim = int(np.cbrt(xyz_eval_numpy.shape[0]))  # Assuming a cubic grid
    x_eval, y_eval, z_eval = xyz_eval_numpy[:, 0].reshape(grid_dim, grid_dim, grid_dim), xyz_eval_numpy[:, 1].reshape(grid_dim, grid_dim, grid_dim), xyz_eval_numpy[:, 2].reshape(grid_dim, grid_dim, grid_dim)

    if eval_nn_at_train:
        u_pred_tensor = model(xyz_eval)  # Evaluate the model
    else:
        u_pred_tensor = model(xyz_eval)  # Evaluate the model on eval data if different from training data

    u_pred_numpy = u_pred_tensor.detach().numpy().reshape(grid_dim, grid_dim, grid_dim)

    # Configure Mayavi figure
    fig = mlab.figure(bgcolor=(1, 1, 1), size=(800, 800))

    # Volume rendering of the predicted solution
    src = mlab.pipeline.scalar_field(x_eval, y_eval, z_eval, u_pred_numpy)
    vol = mlab.pipeline.volume(src, vmin=u_pred_numpy.min(), vmax=u_pred_numpy.max())
    vol._volume_property.set_scalar_opacity_unit_distance(0.1)  # Smaller values give higher opacity

    mlab.title('Neural Network Solution', color=(0, 0, 0))
    mlab.orientation_axes()

    # Display the colorbar and set its properties
    mlab.colorbar(object=vol, title='Value', orientation='vertical', label_fmt='%.2f')
    mlab.outline(color=(0, 0, 0))

    # Render the exact solution if a function is provided
    if exact_sol_func is not None:
        u_exact_numpy = exact_sol_func(xyz_eval_numpy).reshape(grid_dim, grid_dim, grid_dim)
        mlab.figure(bgcolor=(1, 1, 1), size=(800, 800))
        src2 = mlab.pipeline.scalar_field(x_eval, y_eval, z_eval, u_exact_numpy)
        vol2 = mlab.pipeline.volume(src2, vmin=u_exact_numpy.min(), vmax=u_exact_numpy.max())
        mlab.title('Exact Solution', color=(0, 0, 0))
        mlab.orientation_axes()
        mlab.colorbar(object=vol2, title='Value', orientation='vertical', label_fmt='%.2f')
        mlab.outline(color=(0, 0, 0))

    mlab.show()

def plot_volume_rendering_pyvista(model, xyz_eval, opacity_str, eval_nn_at_train=False, exact_sol_func=None, save_fig=False):
    # Formatting specs
    pv_font_size = 10
    pv_font_family = 'times' # times, arial, courier
    pv_bar_width = 0.5
    pv_bar_position_x = (1 - pv_bar_width) / 2
    pv_n_labels = 5
    pv_title_position = 'upper_edge'
    pv_title_font_size = 12
    scalar_bar_dict = {'title': '', 'width': pv_bar_width, 'position_x': pv_bar_position_x, 'label_font_size': pv_font_size, 'font_family': pv_font_family, 'n_labels': pv_n_labels, 'use_opacity': False, 'vertical': False}

    xyz_eval_numpy = xyz_eval.detach().numpy()
    grid_dim = int(np.cbrt(xyz_eval_numpy.shape[0]))
    x_eval, y_eval, z_eval = np.meshgrid(
        np.linspace(0, 1, grid_dim),
        np.linspace(0, 1, grid_dim),
        np.linspace(0, 1, grid_dim),
        indexing='ij'
    )

    if eval_nn_at_train:
        u_pred_tensor = model(xyz_eval)  # Evaluate the model
    else:
        u_pred_tensor = model(xyz_eval)  # Evaluate the model on eval data if different from training data

    u_pred_numpy = u_pred_tensor.detach().numpy().reshape(grid_dim, grid_dim, grid_dim)

    # Setup the ImageData grid for NN prediction
    grid = pv.ImageData(spacing=(1 / (grid_dim - 1), 1 / (grid_dim - 1), 1 / (grid_dim - 1)),
                        origin=(0, 0, 0), dimensions=(grid_dim, grid_dim, grid_dim))
    grid.point_data['values'] = u_pred_numpy.flatten(order='F')  # Assign values to the grid points

    # Setup plotter for NN prediction
    plotter_nn = pv.Plotter(window_size=(48*FIGSIZE[0], 96*FIGSIZE[1])) # sizes in pixels
    plotter_nn.add_volume(grid, scalars='values', cmap='inferno', opacity=opacity_str, scalar_bar_args=scalar_bar_dict)
    plotter_nn.add_text("NN Prediction", font_size=pv_title_font_size, font=pv_font_family, position=pv_title_position)
    plotter_nn.show()

    if exact_sol_func is not None:
        u_exact_numpy = exact_sol_func(xyz_eval_numpy).reshape(grid_dim, grid_dim, grid_dim)
        grid_exact = pv.ImageData(spacing=(1 / (grid_dim - 1), 1 / (grid_dim - 1), 1 / (grid_dim - 1)),
                                  origin=(0, 0, 0), dimensions=(grid_dim, grid_dim, grid_dim))
        grid_exact.point_data['values'] = u_exact_numpy.flatten(order='F')

        # Setup plotter for exact solution
        plotter_exact = pv.Plotter(window_size=(48*FIGSIZE[0], 96*FIGSIZE[1])) # sizes in pixels
        plotter_exact.add_volume(grid_exact, scalars='values', cmap='inferno', opacity=opacity_str, scalar_bar_args=scalar_bar_dict)
        plotter_exact.add_text("Exact Solution", font_size=pv_title_font_size, font=pv_font_family, position=pv_title_position)
        plotter_exact.show()

def plot_isosurface_pyvista(model, xyz_eval, level, exact_sol_func=None):
    # Ensure the input tensor does not require gradient computation
    xyz_eval = xyz_eval.detach()

    # Predict using the neural network
    u_pred = model(xyz_eval).detach().reshape(-1).numpy()

    # Convert the XYZ coordinates and predictions into a PyVista grid
    grid_dim = int(np.cbrt(len(xyz_eval)))  # Assume cubic domain for reshaping
    xyz_eval_numpy = xyz_eval.numpy().reshape(grid_dim, grid_dim, grid_dim, 3)
    grid = pv.StructuredGrid(xyz_eval_numpy[:,:,:,0], xyz_eval_numpy[:,:,:,1], xyz_eval_numpy[:,:,:,2])
    grid["values"] = u_pred

    # Create a plotter and add the neural network solution as an isosurface
    plotter = pv.Plotter()
    plotter.add_mesh(grid.contour(isosurfaces=level, scalars="values"),
                     opacity=0.5, color="red", name="NN Solution")

    # If an exact solution function is provided, plot it as well
    if exact_sol_func:
        u_exact = exact_sol_func(xyz_eval).detach().numpy()
        grid["exact_values"] = u_exact
        plotter.add_mesh(grid.contour(isosurfaces=level, scalars="exact_values"),
                         opacity=0.5, color="blue", name="Exact Solution")

    # Add labels and show the plot
    plotter.add_axes()
    plotter.add_title("Isosurface of Neural Network and Exact Solutions")
    plotter.show()

def plot_loss_vs_epoch(loss_values):
    plt.figure(figsize=FIGSIZE)
    plt.plot(loss_values, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.yscale('log')  # Set the y-axis to logarithmic scale
    plt.legend()
    plt.show()

def plot_pde_residuals(model, bvp, xyz_train_tensor):
    # Predictions from the neural network
    u_pred = model(xyz_train_tensor)

    # Make sure predictions are detached and require gradients for further computation
    u_pred.requires_grad_(True)

    # Compute derivatives with respect to both dimensions
    grad_outputs = torch.ones_like(u_pred)
    grads = torch.autograd.grad(u_pred, xyz_train_tensor, grad_outputs=grad_outputs, create_graph=True)[0]
    u_x, u_y = grads[:, 0].view(-1, 1), grads[:, 1].view(-1, 1)
    u_xx = torch.autograd.grad(u_x, xyz_train_tensor, grad_outputs=grad_outputs, create_graph=True)[0][:, 0]
    u_yy = torch.autograd.grad(u_y, xyz_train_tensor, grad_outputs=grad_outputs, create_graph=True)[0][:, 1]

    # Evaluate the PDE residuals
    residuals = np.abs(bvp.eval_pde(xyz_train_tensor, u_pred, u_x, u_y, u_xx, u_yy).detach().numpy())

    # Reshape for plotting
    num_points_per_dim = int(np.sqrt(xyz_train_tensor.shape[0]))
    x = xyz_train_tensor[:, 0].view(num_points_per_dim, num_points_per_dim).detach().numpy()
    y = xyz_train_tensor[:, 1].view(num_points_per_dim, num_points_per_dim).detach().numpy()
    residuals_reshaped = residuals.reshape(num_points_per_dim, num_points_per_dim)

    # Plotting
    plt.figure(figsize=FIGSIZE)
    contour = plt.contourf(x, y, residuals_reshaped, cmap='viridis')
    plt.colorbar(contour)
    plt.title('PDE Residuals (abs) Across the Domain')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# MESH GENERATION
def random_mesh(domain_bounds, num_interior_points, x_bound_points, y_bound_points, z_bound_points):
    # Calculate the scaling and offset for the interior points
    x_range = domain_bounds['x'][1] - domain_bounds['x'][0]
    y_range = domain_bounds['y'][1] - domain_bounds['y'][0]
    z_range = domain_bounds['z'][1] - domain_bounds['z'][0]
    x_offset = domain_bounds['x'][0]
    y_offset = domain_bounds['y'][0]
    z_offset = domain_bounds['z'][0]

    # Interior points
    interior = torch.rand(num_interior_points, 3)
    interior[:, 0] = interior[:, 0] * x_range + x_offset
    interior[:, 1] = interior[:, 1] * y_range + y_offset
    interior[:, 2] = interior[:, 2] * z_range + z_offset

    # Boundary points
    x_edges = torch.linspace(domain_bounds['x'][0], domain_bounds['x'][1], steps=x_bound_points)
    y_edges = torch.linspace(domain_bounds['y'][0], domain_bounds['y'][1], steps=y_bound_points)
    z_edges = torch.linspace(domain_bounds['z'][0], domain_bounds['z'][1], steps=z_bound_points)
    # Create grid points for each face of the cube
    xy_plane_z0 = torch.stack(torch.meshgrid(x_edges, y_edges, indexing='ij'), dim=-1).reshape(-1, 2)
    xy_plane_z1 = xy_plane_z0.clone()
    xy_plane_z0 = torch.cat([xy_plane_z0, torch.full_like(xy_plane_z0[:, :1], domain_bounds['z'][0])], dim=1)
    xy_plane_z1 = torch.cat([xy_plane_z1, torch.full_like(xy_plane_z1[:, :1], domain_bounds['z'][1])], dim=1)

    xz_plane_y0 = torch.stack(torch.meshgrid(x_edges, z_edges, indexing='ij'), dim=-1).reshape(-1, 2)
    xz_plane_y1 = xz_plane_y0.clone()
    xz_plane_y0 = torch.cat([xz_plane_y0[:, :1], torch.full_like(xz_plane_y0[:, :1], domain_bounds['y'][0]), xz_plane_y0[:, 1:]], dim=1)
    xz_plane_y1 = torch.cat([xz_plane_y1[:, :1], torch.full_like(xz_plane_y1[:, :1], domain_bounds['y'][1]), xz_plane_y1[:, 1:]], dim=1)

    yz_plane_x0 = torch.stack(torch.meshgrid(y_edges, z_edges, indexing='ij'), dim=-1).reshape(-1, 2)
    yz_plane_x1 = yz_plane_x0.clone()
    yz_plane_x0 = torch.cat([torch.full_like(yz_plane_x0[:, :1], domain_bounds['x'][0]), yz_plane_x0], dim=1)
    yz_plane_x1 = torch.cat([torch.full_like(yz_plane_x1[:, :1], domain_bounds['x'][1]), yz_plane_x1], dim=1)

    # Combine all boundary points
    boundary = torch.cat([xy_plane_z0, xy_plane_z1, xz_plane_y0, xz_plane_y1, yz_plane_x0, yz_plane_x1], dim=0)

    # Combine interior and boundary points
    xyz_train = torch.cat([interior, boundary], dim=0)
    xyz_train.requires_grad_(True)
    return xyz_train

def uniform_mesh(domain_bounds, x_points, y_points, z_points):
    # Generate linearly spaced points for each dimension
    x_points = torch.linspace(domain_bounds['x'][0], domain_bounds['x'][1], steps=x_points)
    y_points = torch.linspace(domain_bounds['y'][0], domain_bounds['y'][1], steps=y_points)
    z_points = torch.linspace(domain_bounds['z'][0], domain_bounds['z'][1], steps=z_points)

    # Create a mesh grid for 3D space
    x_grid, y_grid, z_grid = torch.meshgrid(x_points, y_points, z_points, indexing='ij')  # Adjust for 3D

    # Flatten and stack to create 3D points
    xyz_train = torch.stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()], dim=1)

    xyz_train.requires_grad_(True)  # Enable gradient tracking

    return xyz_train


####################################################################################################
####################################################################################################
####################################################################################################

BVP_NO = 2
BAR_APPROACH = True
OPTIMISER_NAME = 'lbfgs' # lbfgs, adam
NO_POINTS_DIR = 25
MESH_TYPE = 'uniform' # uniform, random

hidden_units = 50
depth = 1

SAVE_FIGURE = False

if BVP_NO == 0:
    # Laplace's equation, TRIVIAL solution
    def PDE_func(xyz, u, u_x, u_y, u_z, u_xx, u_yy, u_zz):
        return torch.squeeze(u_xx + u_yy + u_zz)

    def boundary_east(x, z):
        return 0.0
    def boundary_west(x, z):
        return 0.0
    def boundary_north(y, z):
        return 0.0
    def boundary_south(y, z):
        return 0.0
    def boundary_bottom(x, y):
        return 0.0
    def boundary_top(x, y):
        return 0.0

    # Domain bounds
    domain_bounds = {'x': (0, 1), 'y': (0, 1), 'z': (0, 1)}

    # Function satisfying boundary conditions
    def g_func(xyz):
        return torch.zeros(xyz.size(0))

    def exact_sol(xyz):
        # Since the boundary conditions and the PDE suggest a trivial solution:
        return torch.zeros(xyz.shape[0], 1)

    no_epochs = 30
    learning_rate = 0.05

    gamma=10
if BVP_NO == 1:
    # Laplace's equation
    def PDE_func(xyz, u, u_x, u_y, u_z, u_xx, u_yy, u_zz):
        return torch.squeeze(u_xx + u_yy + u_zz) + (3 * (np.pi**2) * torch.sin(np.pi * torch.squeeze(xyz[:,0])) * torch.sin(np.pi * torch.squeeze(xyz[:,1])) * torch.sin(np.pi * torch.squeeze(xyz[:,2])))

    def boundary_east(y, z):
        return 0.0
    def boundary_west(y, z):
        return 0.0
    def boundary_north(x, z):
        return 0.0
    def boundary_south(x, z):
        return 0.0
    def boundary_bottom(x, y):
        return 0.0
    def boundary_top(x, y):
        return 0.0

    # Domain bounds
    domain_bounds = {'x': (0, 1), 'y': (0, 1), 'z': (0, 1)}

    # Function satisfying boundary conditions
    def g_func(xyz):
        return torch.zeros(xyz.size(0))

    def exact_sol(xyz):
        return np.sin(np.pi * xyz[:,0]) * np.sin(np.pi * xyz[:,1]) * np.sin(np.pi * xyz[:,2])

    no_epochs = 300
    learning_rate = 0.2

    gamma = 100

    # For plotting isosurfaces
    level = 0.5
if BVP_NO == 2:
    # Laplace's equation, higher frequency (modest for 3D...)!

    def PDE_func(xyz, u, u_x, u_y, u_z, u_xx, u_yy, u_zz):
        return torch.squeeze(u_xx + u_yy + u_zz) + (3 * (2 * np.pi)**2 * torch.sin(2 * np.pi * torch.squeeze(xyz[:,0])) * torch.sin(2 * np.pi * torch.squeeze(xyz[:,1])) * torch.sin(2 * np.pi * torch.squeeze(xyz[:,2])))

    def boundary_east(y, z):
        return 0.0
    def boundary_west(y, z):
        return 0.0
    def boundary_north(x, z):
        return 0.0
    def boundary_south(x, z):
        return 0.0
    def boundary_bottom(x, y):
        return 0.0
    def boundary_top(x, y):
        return 0.0

    # Domain bounds
    domain_bounds = {'x': (0, 1), 'y': (0, 1), 'z': (0, 1)}

    # Function satisfying boundary conditions
    def g_func(xyz):
        return torch.zeros(xyz.size(0))

    def exact_sol(xyz):
        return np.sin(2 * np.pi * xyz[:,0]) * np.sin(2 * np.pi * xyz[:,1]) * np.sin(2 * np.pi * xyz[:,2])

    # no_epochs = 2000 # with Adam
    no_epochs = 10 # with LBFGS
    
    # learning_rate = 0.12 # with Adam
    learning_rate = 0.1 # with LBFGS

    gamma = 100
    
    # opacity_str = 'sigmoid_1' # map from -1 to 1
    opacity_str = 0.08
if BVP_NO == 3:
    # Laplace's equation
    def PDE_func(xyz, u, u_x, u_y, u_xx, u_yy):
        x, y = torch.squeeze(xyz[:,0]), torch.squeeze(xyz[:,1])
        return torch.squeeze(u_xx + u_yy) - torch.squeeze(2 * x * (y - 1) * (y - 2*x + x*y + 2) * torch.exp(x - y))

    def boundary_east(y):
        return 0.0
    def boundary_west(y):
        return 0.0
    def boundary_north(x):
        return 0.0
    def boundary_south(x):
        return 0.0 

    # Domain bounds
    domain_bounds = {'x': (0, 1), 'y': (0, 1)}

    # Function satisfying boundary conditions
    def g_func(xyz):
        return torch.zeros(xyz.size(0))

    def exact_sol(xyz):
        x, y = xyz[:,0], xyz[:,1]
        return np.exp(x - y) * x * (1 - x) * y * (1 - y)

    no_epochs = 500
    learning_rate = 0.1

    gamma = 10
if BVP_NO == 4:
    # Laplace's equation, linear solution
    def PDE_func(xyz, u, u_x, u_y, u_z, u_xx, u_yy, u_zz):
        return torch.squeeze(u_xx + u_yy + u_zz)

    def boundary_east(y, z):
        return 1
    def boundary_west(y, z):
        return 0
    def boundary_north(x, z):
        return x
    def boundary_south(x, z):
        return x
    def boundary_bottom(x, y):
        return x
    def boundary_top(x, y):
        return x

    # Domain bounds
    domain_bounds = {'x': (0, 1), 'y': (0, 1), 'z': (0, 1)}

    # Function satisfying boundary conditions
    def g_func(xyz):
        x, y, z = torch.squeeze(xyz[:,0]), torch.squeeze(xyz[:,1]), torch.squeeze(xyz[:,2])
        return x

    def exact_sol(xyz):
        x, y = xyz[:,0], xyz[:,1]
        return x

    no_epochs = 100
    learning_rate = 0.1

    gamma = 20
if BVP_NO == 5:
    # Laplace's equation
    def PDE_func(xyz, u, u_x, u_y, u_xx, u_yy):
        return u_xx + u_yy

    def boundary_east(y):
        return 1 - y**2
    def boundary_west(y):
        return - y**2
    def boundary_north(x):
        return x**2 - 1
    def boundary_south(x):
        return x**2

    # Domain bounds
    domain_bounds = {'x': (0, 1), 'y': (0, 1)}

    # Function satisfying boundary conditions
    def g_func(xyz):
        Exception('not in use')

    def exact_sol(xyz):
        return xyz[:,0] ** 2 - xyz[:,1] ** 2

    no_epochs = 5000
    learning_rate = 0.03

    gamma = 100

# INFORMATIVE FILE NAME FOR SAVING
file_name = f'problem{str(BVP_NO)}-depth{depth}-width{hidden_units}-bar{BAR_APPROACH}-mesh{MESH_TYPE}-points{NO_POINTS_DIR}-optimiser{OPTIMISER_NAME}-epochs{no_epochs}-lr{learning_rate}-gamma{gamma}'

# STILL NEED TO APPEND TYPE OF PLOT TO THE END OF THIS STRING!
base_plot_path = plot_file_base + file_name

# Boundary conditions dictionary
bcs = {
    'east':   boundary_east,
    'north':  boundary_north,
    'west':   boundary_west,
    'south':  boundary_south,
    'bottom': boundary_bottom,
    'top':    boundary_top
}

# BVP instance
bvp = BVP3D(PDE_func=PDE_func, domain_bounds=domain_bounds, bcs=bcs, g_func=g_func)

# Create the neural network
model = NeuralNetwork3D(bvp, hidden_units=hidden_units, depth=depth, bar_approach=BAR_APPROACH)

# Optimizer
if OPTIMISER_NAME == 'lbfgs':
    optimiser = torch.optim.LBFGS(model.parameters(), lr=learning_rate)
elif OPTIMISER_NAME == 'adam':
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Loss class instance
loss_class = CustomLoss(bvp, gamma=gamma, bar_approach=BAR_APPROACH)

# GENERATE MESHES
if MESH_TYPE == 'uniform':
    xyz_train = uniform_mesh(domain_bounds, NO_POINTS_DIR, NO_POINTS_DIR, NO_POINTS_DIR)
elif MESH_TYPE == 'random':
    xyz_train = random_mesh(domain_bounds, NO_POINTS_DIR, 50, 50)

xyz_eval  = uniform_mesh(domain_bounds, 50, 50, 50)

# Training the model
loss_values = train_model(model, optimiser, bvp, loss_class, xyz_train, no_epochs)

# PLOTTING
# plot_loss_vs_epoch(loss_values)
# plot_pde_residuals(model, bvp, xyz_train)

# from mpl_toolkits.mplot3d import Axes3D
# from mayavi import mlab
# plot_isosurface_mlab(model, xyz_eval, level=level, exact_sol_func=exact_sol)
# plot_volume_rendering_mlab(model, xyz_eval, eval_nn_at_train=False, exact_sol_func=exact_sol)

import pyvista as pv
# plot_isosurface_pyvista(model, xyz_eval, level=[0.2, 0.5], exact_sol_func=exact_sol)
plot_volume_rendering_pyvista(model, xyz_eval, eval_nn_at_train=False, exact_sol_func=exact_sol, opacity_str=opacity_str, save_fig=SAVE_FIGURE)