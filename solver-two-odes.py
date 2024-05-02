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

# DEPRECATED BVP class
class BVP:
    def __init__(self, alphas, ns, g_func, domain_ends, bcs):
        """
        Initializes a boundary value problem for a second-order ODE.
        Args:
            alphas (torch.Tensor): A tensor of shape (num_eqs, num_funcs, 3) containing the coefficients 
                                   for each term in each ODE (i.e., alpha0, alpha1, alpha2 for y, y', y'').
            ns (torch.Tensor): A tensor of shape (num_eqs, num_funcs, 3) containing the powers 
                               for each term in each ODE.
            g_func (callable): The right-hand side function g(x) which should return a torch tensor of values.
            domain_ends (tuple): The domain ends (a, b).
            bcs (dict): Boundary conditions with details for each end ('a' and 'b') and possibly for each variable.

        """
        
        # The bcs dictionary's form has a lot of info about the BVP
        self.bcs = bcs
        
        # Single ODE case
        if isinstance(bcs['a'], tuple):
            self.dim = 1
            self.alpha0, self.alpha1, self.alpha2 = alphas
            self.n0, self.n1, self.n2 = ns
        else:
            # System of self.dim ODEs
            self.alphas = alphas # Should be a 3D array. Indexing goes as [eqn #, unknown funcn #, order of derivative]
            self.dim = alphas.size(0)
            
        
        # g_func should be multi-dimensional whenever appropriate
        self.g_func = g_func

        # this is always simple
        self.domain_ends = domain_ends

    # y-related inputs should be vectors as appropriate in ODE system cases
    def eval_ode(self, x, y, y_x, y_xx):
        """
        Evaluates simple residual.

        Args:
            x (torch.Tensor): Input tensor of independent variable values.
            y (torch.Tensor): Tensor of shape (num_funcs,) of function values at x.
            y_x (torch.Tensor): Tensor of shape (num_funcs,) of first derivatives at x.
            y_xx (torch.Tensor): Tensor of shape (num_funcs,) of second derivatives at x.
        
        Returns (in system case):
            torch.Tensor: A tensor representing the residuals for the system of ODEs.
        """
        if self.dim == 1:
            term2 = self.alpha2 * (y_xx ** self.n2)
            term1 = self.alpha1 * (y_x ** self.n1)
            term0 = self.alpha0 * (y ** self.n0)
            return term2 + term1 + term0 - self.g_func(x)
        else:
            derivatives = torch.stack([y, y_x, y_xx], dim=0)  # Shape (3, num_funcs)

            # Compute the terms for each equation using broadcasting:
            # alphas.shape == (num_eqs, num_funcs, 3)
            # derivatives.shape == (3, num_funcs)
            # powers.shape == (num_eqs, num_funcs, 3)
            terms = self.alphas * derivatives.unsqueeze(0) ** self.ns  # Applying power element-wise

            # Sum over the last dimension to compute the ODE left-hand sides
            ode_lhs = terms.sum(dim=-1)  # Sum across all derivatives, shape (num_eqs, num_funcs)
            
            # Subtract the right-hand side (g(x)) from the ODE left-hand sides
            residuals = ode_lhs - self.g_func(x)

            return residuals

class BVPflexible:
    def __init__(self, ODE_funcs, domain_ends, bcs):
        """
        Initializes a boundary value problem for a system of second-order ODEs.

        Args:
            terms_funcs (list of callables): Each item corresponds to an ODE and contains a callables that returns the residual.
            domain_ends (tuple): The domain ends (a, b).
            bcs (dictionary): Boundary conditions with details for each end ('a' and 'b') and possibly for each variable.
        """
        
        # The bcs dictionary's form has a lot of info about the BVP
        self.ODE_funcs = ODE_funcs
        self.domain_ends = domain_ends
        self.bcs = bcs
        self.dim = len(self.ODE_funcs) # dimension of system

    # y-related inputs should be vectors as appropriate in ODE system cases
    def eval_ode(self, x, u, u_x, u_xx, v, v_x, v_xx):
        """
        Evaluates residual.

        Args:
            x (torch.Tensor): Input tensor of independent variable values.
            y (torch.Tensor): Tensor of shape (num_points, num_funcs) of function values at x.
            y_x (torch.Tensor): Tensor of shape (num_points, num_funcs) of first derivatives at x.
            y_xx (torch.Tensor): Tensor of shape (num_points, num_funcs) of second derivatives at x.
        
        Returns:
            torch.Tensor: A tensor representing the residuals for the system of ODEs.
        """

        lhs_u = ODE_funcs[0]
        lhs_v = ODE_funcs[1]
        
        # Each of these a 1D tensor
        residuals_u = lhs_u(x, u, u_x, u_xx, v, v_x, v_xx)
        residuals_v = lhs_v(x, u, u_x, u_xx, v, v_x, v_xx)
        
        # Put into 1D tensor
        residuals = torch.cat((residuals_u, residuals_v), 1)
        return residuals


    
class NeuralNetwork(nn.Module):
    def __init__(self, bvp, input_features=1, output_features=1, hidden_units=5, depth=1, bar_approach=False):
        """
        Initializes all required hyperparameters for a typical model and dynamically
        creates layers based on the depth of the network.

        Args:
            input_features (int): Number of input features to the model.
            output_features (int): Number of output features of the model (how many classes there are).
            hidden_units (int): Number of hidden units between layers, default 5.
            depth (int): Determines the depth of the network, default 1.
        """
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

    def y_bar_scaling(self, x, y_hat):
        bcs = self.bvp.bcs
        a, b = self.bvp.domain_ends
        a, b = torch.tensor([float(a)], requires_grad=True), torch.tensor([float(b)], requires_grad=True) # make tensor, for taking derivatives

        # Initialise
        y_bar = torch.zeros_like(y_hat)

        # Note: For an n-system, bcs should be an n-tuple, where each element is a tuple ((bc_type_a, bc_value_a), (bc_type_b, bc_value_b))
        for idx in range(self.bvp.dim):
            bc_type_a, bc_value_a = bcs[idx][0]
            bc_type_b, bc_value_b = bcs[idx][1]

            if bc_type_a == 'dirichlet' and bc_type_b == 'dirichlet':
                # 1st scaling option (seems to work well)
                # y_bar = (x - a) * (b - x) * y_hat + (x - a)/(b - a) * y_b + (b - x) / (b - a) * y_a
                y_bar[:, idx] = torch.squeeze((x - a) * (b - x)) * y_hat[:, idx] + torch.squeeze((x - a) / (b - a) * bc_value_b) + torch.squeeze((b - x) / (b - a) * bc_value_a)
                
                # 2nd scaling option (seems to work worse)
                # y_bar = y_hat + (b - x)*(y_a - y_hat[0])/(b - a) + (x - a)*(y_b - y_hat[-1])/(b - a)
            elif bc_type_a == 'dirichlet' and bc_type_b == 'neumann':
                # 1st scaling approach (Kathryn-proposed) (I think I copied down wrong, does not enforce BC)
                # y_bar[:, idx] = y_b * x + y_a - a * y_b + (x - a) * (y_hat[:, idx] - y_hat_b - y_hat_x_b) / (b - a)
                
                # 2nd scaling approach (Maria-proposed)
                # y_bar[:, idx] = y_hat[:, idx] + (y_a - y_hat[0, idx]) + (x - a) * (y_b - y_hat_x_b)

                y_hat_b = self.stack(b)[idx]
                y_hat_x_b = torch.autograd.grad(y_hat_b, b, torch.ones_like(y_hat_b), create_graph=True)[0]
                # MY FIX to Kathryn proposal
                y_bar[:, idx] = torch.squeeze(bc_value_b * x) + bc_value_a - a * bc_value_b + torch.squeeze(x - a) * ((y_hat[:, idx] - y_hat_b) / (b - a) - y_hat_x_b)

            elif bc_type_a == 'neumann' and bc_type_b == 'dirichlet':
                
                # 1st scaling approach (Kathryn-proposed) (I think I copied down wrong, does not enforce BC)
                # y_bar = y_a * x + y_b - b * y_a + (x - b) * (y_hat - y_hat_a - y_hat_x_a) / (b - a)

                # MY FIX to Kathryn proposal
                # y_bar = y_a * x + y_b - b * y_a + (b - x) * ((y_hat - y_hat_a)/(b - a) - y_hat_x_a)

                # 2nd scaling approach (Maria-proposed)
                # y_bar = y_hat + (y_b - y_hat[-1]) + (x - b) * (y_a - y_hat_x_a)

                y_hat_a = self.stack(a)[idx]
                y_hat_x_a = torch.autograd.grad(y_hat_a, a, torch.ones_like(y_hat_a), create_graph=True)[0]
                y_bar[:, idx] = torch.squeeze(bc_value_a * x) + bc_value_b - b * bc_value_a + torch.squeeze(b - x) * ((y_hat[:, idx] - y_hat_a) / (b - a) - y_hat_x_a)

        return y_bar



    def forward(self, x):
        if self.bar_approach:
            y_hat = self.stack(x)
            y_bar = self.y_bar_scaling(x, y_hat)
            return y_bar
        else:
            return self.stack(x)
    
class CustomLoss(nn.Module):
    def __init__(self, bvp, gamma, bar_approach=False):
        super().__init__()
        self.bvp = bvp
        self.gamma = gamma
        self.bar_approach = bar_approach

    def forward(self, x, u, v):
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]

        # Compute the ODE residuals and their mean squared error
        ode_loss = torch.mean((self.bvp.eval_ode(x, u, u_x, u_xx, v, v_x, v_xx) ** 2))

        if self.bar_approach:
            return ode_loss
        else:
            # Calculate boundary condition losses for each equation
            # Recall bcs are indexed by [eqn no.][a or b][type or value]
            u_a, u_b = self.bvp.bcs[0][0][1], self.bvp.bcs[0][1][1]
            v_a, v_b = self.bvp.bcs[1][0][1], self.bvp.bcs[1][1][1]

            bc_loss = self.gamma * torch.sum((u[0] - u_a)**2 + (u[-1] - u_b)**2 + (v[0] - v_a)**2 + (v[-1] - v_b)**2)
            return ode_loss + bc_loss
    
def train_model(model, optimiser, bvp, loss_class, x_train, no_epochs):
    loss_values = []  # List to store loss values

    for epoch in range(no_epochs):

        # Differentiate the optimisation process based on the optimiser type
        if isinstance(optimiser, torch.optim.LBFGS):
            # Define the closure function for LBFGS
            def closure():
                optimiser.zero_grad()

                y_pred = model(x_train)
                # y_pred.requires_grad_(True)
                loss = loss_class.forward(x_train, y_pred)
                loss.backward()
                return loss
            
            # Step through the optimiser
            loss = optimiser.step(closure)
        else: # For first-order optimisers like Adam or SGD
            optimiser.zero_grad()
            y_pred = model(x_train)
            y_pred.requires_grad_(True)
            loss = loss_class(x_train, y_pred)

            loss.backward()
            optimiser.step()

        # Store the loss value for this epoch
        loss_values.append(loss.item())

        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss.item():e}")

    return loss_values  # Return the list of loss values

def plot_loss_vs_epoch(loss_values):
    plt.figure(figsize=FIGSIZE)
    plt.plot(loss_values, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.yscale('log')  # Set the y-axis to logarithmic scale
    plt.legend()
    plt.show()

def plot_predictions(model, x_train_tensor, exact_sol_func=None):
    # Convert the training tensor to numpy for plotting
    x_train_numpy = x_train_tensor.detach().numpy().flatten()
    
    # Predictions from the neural network
    # We DO need gradients sometimes for evaluation (bar approach with Neumann conditions, etc.)
    y_pred_tensor = model(x_train_tensor)
    y_pred_numpy = y_pred_tensor.detach().numpy()
    
    num_equations = y_pred_numpy.shape[1]
    
    # Set up the figure with multiple subplots
    fig, axes = plt.subplots(num_equations, 1, figsize=FIGSIZE)
    
    if num_equations == 1:
        axes = [axes]  # make it iterable if only one plot
    
    # Plot predictions for each equation
    for i in range(num_equations):
        axes[i].plot(x_train_numpy, y_pred_numpy[:, i], label=f'NN Predictions (eq {i+1})', color='r', linestyle='--', marker='o')
        if exact_sol_func is not None:
            y_exact_numpy = exact_sol_func(x_train_numpy)[i]
            axes[i].plot(x_train_numpy, y_exact_numpy, label=f'Analytical Solution (eq {i+1})', color='b', linestyle='-')
        
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        axes[i].legend()
        axes[i].set_title(f'NN Predictions vs Exact Solution (Eq {i+1})')

    plt.tight_layout()
    plt.show()

def plot_ode_residuals(model, bvp, x_train_tensor):
    # Convert the training tensor to numpy for x-axis plotting
    x_train_numpy = x_train_tensor.detach().numpy().flatten()
    
    
    # Predictions from the neural network

    y_pred = model(x_train_tensor)

    # y_pred_numpy = y_pred.detach().numpy().flatten()

    # Compute derivatives
    y_pred.requires_grad_(True)
    y_x = torch.autograd.grad(y_pred, x_train_tensor, torch.ones_like(y_pred), create_graph=True)[0]
    y_xx = torch.autograd.grad(y_x, x_train_tensor, torch.ones_like(y_x), create_graph=True)[0]
    
    # Evaluate the ODE residuals
    residuals = bvp.eval_ode(x_train_tensor, y_pred, y_x, y_xx).detach().numpy().flatten() ** 2
    
    # Plotting
    plt.figure(figsize=FIGSIZE)
    plt.plot(x_train_numpy, residuals, label='ODE Residuals', color='blue', linestyle='-.')
    plt.axhline(0, color='black', lw=1)  # Zero line for reference
    plt.xlabel('x')
    plt.ylabel('Residual (abs value)')
    plt.yscale('log')
    plt.title('ODE Residuals Across the Domain')
    plt.legend()
    plt.show()


"""

ENTERING RELEVANT PARAMETERS

"""

BVP_NO = 10
BAR_APPROACH = False

if BVP_NO == 10:
    # KATHRYN PROPOSED SYSTEM for u(x), v(x)
    # neumann + dirichlet conditions
    domain_ends = (0, 1)
    bcs = (
        (('dirichlet', 1), ('dirichlet', 1)),
        (('dirichlet', 0), ('dirichlet', 0)),
    )

    def eqn1(x, u, u_x, u_xx, v, v_x, v_xx):
        return -y[:,0] + torch.squeeze(x) * y[:,0] - y[:,1] - 2 - torch.squeeze(x)
    
    def eqn2(x, u, u_x, u_xx, v, v_x, v_xx):
        return - torch.squeeze(y_xx[:,1]) + y[:,1] - 6 * torch.squeeze(x) + 2 - torch.squeeze(torch.squeeze(x) ** 2) * (1 - torch.squeeze(x))
    
    ODE_funcs = [eqn1,
                 eqn2]

    def exact_sol(x):
        return np.array([1 + x * (1 - x),
                         x**2 * (1 - x)])

    no_epochs = 10000
    learning_rate = 0.02
elif BVP_NO == 11:
    # proof of concept for systems solver. UNCOUPLED equations
    domain_ends = (0, 1)
    bcs = (
        (('dirichlet', 1), ('dirichlet', 1)),
        (('dirichlet', 0), ('dirichlet', 0)),
    )

    # simple parabolic solution
    def eqn1(x, y, y_x, y_xx):
        return torch.squeeze(y_xx[:,0]) - torch.squeeze(y[:,0]**2) + torch.squeeze(3 + 2 * x - x ** 2 - 2 * x ** 3 + x ** 4)
    
    # simple boundary layer
    def eqn2(x, y, y_x, y_xx):
        return torch.squeeze(y_x[:,1]) - torch.squeeze(0.03 * y_xx[:,1]) - 1
    
    ODE_funcs = [eqn1,
                 eqn2]

    def exact_sol(x):
        return np.array([1 + x * (1 - x),
                         0*x]) # in reality don't know

    no_epochs = 15000
    learning_rate = 0.003

# Define BVP (routine)
my_bvp = BVPflexible(
    ODE_funcs=ODE_funcs,
    domain_ends=domain_ends,
    bcs=bcs
)

if BVP_NO == 0:
    loss_class = CustomLoss(bvp=my_bvp, gamma=1.5, bar_approach=BAR_APPROACH)
elif BVP_NO == 1:
    loss_class = CustomLoss(bvp=my_bvp, gamma=1.5, bar_approach=BAR_APPROACH)
elif BVP_NO == 2:
    loss_class = CustomLoss(bvp=my_bvp, gamma=200, bar_approach=BAR_APPROACH)
elif BVP_NO == 3:
    loss_class = CustomLoss(bvp=my_bvp, gamma=0.1, bar_approach=BAR_APPROACH)
elif BVP_NO == 4:
    loss_class = CustomLoss(bvp=my_bvp, gamma=1.0, bar_approach=BAR_APPROACH)
elif BVP_NO == 5:
    loss_class = CustomLoss(bvp=my_bvp, gamma=1.5, bar_approach=BAR_APPROACH)
elif BVP_NO == 6:
    loss_class = CustomLoss(bvp=my_bvp, gamma=1.5, bar_approach=BAR_APPROACH)
elif BVP_NO == 7:
    loss_class = CustomLoss(bvp=my_bvp, gamma=1.5, bar_approach=BAR_APPROACH)
elif BVP_NO == 8:
    loss_class = CustomLoss(bvp=my_bvp, gamma=1.5, bar_approach=BAR_APPROACH)
elif BVP_NO == 9:
    loss_class = CustomLoss(bvp=my_bvp, gamma=1.5, bar_approach=BAR_APPROACH)
elif BVP_NO == 10:
    loss_class = CustomLoss(bvp=my_bvp, gamma=5, bar_approach=BAR_APPROACH)
elif BVP_NO == 11:
    loss_class = CustomLoss(bvp=my_bvp, gamma=5, bar_approach=BAR_APPROACH)


# TRAINING POINTS
training_points = np.linspace(my_bvp.domain_ends[0], my_bvp.domain_ends[1], 50)
x_train = torch.tensor(training_points).reshape(len(training_points), 1)
x_train = x_train.to(torch.float32).requires_grad_(True)

# MODEL
ANN_width = 8
ANN_depth = 2

output_features = my_bvp.dim
input_features = 1

model = NeuralNetwork(my_bvp, input_features, output_features, ANN_width, ANN_depth, bar_approach=BAR_APPROACH)

# OPTIMISER
optimiser = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

# Loss
loss_values = train_model(model, optimiser, my_bvp, loss_class, x_train, no_epochs)

# PLOTTING
plot_predictions(model, x_train, exact_sol_func=exact_sol)

plot_loss_vs_epoch(loss_values)

plot_ode_residuals(model, my_bvp, x_train)