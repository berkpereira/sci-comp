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

class BVP:
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
    def eval_ode(self, x, y, y_x, y_xx):
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
        residuals_temp = torch.zeros_like(y)
        
        for idx, lhs in enumerate(self.ODE_funcs):
            residuals_temp[:,idx] = lhs(x, y, y_x, y_xx)
        
        residuals = residuals_temp
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
            elif bc_type_a == 'neumann' and bc_type_b == 'neumann':
                y_hat_a = self.stack(a)[idx]
                y_hat_x_a = torch.autograd.grad(y_hat_a, a, torch.ones_like(y_hat_a), create_graph=True)[0]
                y_hat_b = self.stack(b)[idx]
                y_hat_x_b = torch.autograd.grad(y_hat_b, b, torch.ones_like(y_hat_b), create_graph=True)[0]

                C1 = (bc_value_a + bc_value_b - y_hat_x_a - y_hat_x_b) / 2
                C2 = (bc_value_b - bc_value_a + y_hat_x_a - y_hat_x_b) / (2 * (b - a))
                y_bar[:, idx] = torch.squeeze(C1 * x) + C2 * torch.squeeze(x - a) * torch.squeeze(x - b) + y_hat[:, idx]

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

    def forward(self, x, y):
        # Assuming y shape is [batch_size, num_equations]
        batch_size, num_equations = y.shape

        # Prepare to collect derivatives
        y_x = torch.zeros_like(y)
        y_xx = torch.zeros_like(y)

        # Calculate derivatives individually for each equation to maintain correct shape
        for i in range(num_equations):
            # We need to keep y[:, i:i+1] to keep dimensionality for grad
            y_x[:, i:i+1] = torch.autograd.grad(y[:, i].sum(), x, create_graph=True, allow_unused=True)[0]
            y_xx[:, i:i+1] = torch.autograd.grad(y_x[:, i].sum(), x, create_graph=True, allow_unused=True)[0]

        # Compute the ODE residuals and their mean squared error
        ode_loss = torch.mean(self.bvp.eval_ode(x, y, y_x, y_xx) ** 2)

        if self.bar_approach:
            return ode_loss
        else:
            # Calculate boundary condition losses for each equation
            bc_loss = 0
            for i in range(num_equations):
                for j in range(2):
                    if self.bvp.bcs[i][j][0] == 'dirichlet':
                        y_val = self.bvp.bcs[i][j][1]
                        bc_loss += (y[0 if j == 0 else -1, i] - y_val) ** 2
                    elif self.bvp.bcs[i][j][0] == 'neumann':
                        y_val = self.bvp.bcs[i][j][1]
                        bc_loss += (y_x[0 if j == 0 else -1, i] - y_val) ** 2
                    elif self.bvp.bcs[i][j][0] == 'robin':
                        alpha = self.bvp.bcs[i][j][1] # WE HAVE AN EXTRA ENTRY FOR THESE
                        y_val = self.bvp.bcs[i][j][2] 
                        # NOTICE the assumed form of storage for Robin condition (see Obsidian notes for more details)
                        bc_loss += (y[0 if j == 0 else -1, i] + alpha * y_x[0 if j == 0 else -1, i] - y_val) ** 2
            return ode_loss + self.gamma * bc_loss
    
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

BVP_NO = 4
BAR_APPROACH = True

if BVP_NO == 0:
    # BVP proposed by Kathryn
    # ODE: -y'' + y^2 = g(x)
    # g(x) = 3 + 2 * x - x ** 2 - 2 * x ** 3 + x ** 4
    # y(0) = y(1) = 1

    # Should return a 1D tensor
    # x is 1D tensor
    # y, y_x, y_xx are 2D tensors [num_points, num_unknowns]
    def eqn1(x, y, y_x, y_xx):
        return torch.squeeze(3 + 2 * x - x ** 2 - 2 * x ** 3 + x ** 4) + torch.squeeze(y_xx[:, 0]) - y[:,0]**2
    
    # Each function in this list should return a 1D tensor (length = number of points in x)
    ODE_funcs = [eqn1]

    alphas = (1, 0, -1)
    ns = (2, 1, 1)
    domain_ends = (0, 1)
    bcs = (
        (('dirichlet', 1), ('dirichlet', 1)),
    )
    exact_sol = lambda x: 1 + x * (1 - x)
    
    no_epochs = 300
    learning_rate = 0.004
elif BVP_NO == 1:
    # BVP with boundary layer solution
    g_func=lambda x: 1
    alphas = (0, 1, -0.02)
    ns = (1, 1, 1)
    domain_ends = (0, 1)
    bcs = (0, 0)
    exact_sol = None
    
    no_epochs = 10000
    learning_rate = 0.008
elif BVP_NO == 2:
    alphas = (64, 0, 1)
    ns = (1, 1, 1)
    # Right domain end is about x = 2.55
    domain_ends = (0, 9 * np.pi / 16)
    bcs = (1, 0) # DIRICHLET
    g_func = lambda x: 0
    exact_sol = lambda x: np.cos(8 * x) # UNIQUE solution
    
    no_epochs = 50000
    learning_rate = 0.0025
elif BVP_NO == 3:
    alphas = (1, 0, 0)
    ns = (1, 0, 0)
    domain_ends = (0, 10)
    bcs = (0, np.sin(10) * 10)
    g_func = lambda x: torch.sin(x) * x
    exact_sol = lambda x: np.sin(x) * x

    no_epochs = 20000
    learning_rate = 0.001
elif BVP_NO == 4:
    def eqn1(x, y, y_x, y_xx):
        return torch.squeeze(y_xx[:,0]) + torch.squeeze(x) - y[:,0]
    
    # Each function in this list should return a 1D tensor (length = number of points in x)
    ODE_funcs = [eqn1]

    domain_ends = (0, 1)
    bcs = (
        (('neumann', 1), ('neumann', -2)),
    )

    def exact_sol(x):
        return np.array([-(3 * np.exp(1)/(np.exp(2) - 1))*np.exp(x) - (3 * np.exp(1)/(np.exp(2) - 1))*np.exp(-x) + x])
    
    no_epochs = 500
    learning_rate = 0.02
elif BVP_NO == 5:
    # BVP proposed by Kathryn for y_bar approach
    # ODE: -y'' + y^2 = g(x)
    # g(x) = 3 + 2 * x - x ** 2 - 2 * x ** 3 + x ** 4
    # y(0) = y(1) = 1
    alphas = (0, 0, -1)
    ns = (1, 1, 1)
    domain_ends = (0, 1)
    bcs = (1, 1)
    g_func = lambda x: 2
    exact_sol = lambda x: 1 + x * (1 - x)
    
    no_epochs = 10000
    learning_rate = 0.001
elif BVP_NO == 6:
    # BVP proposed by Kathryn for Neumann BCs
    # ODE: -y'' + y^2 = g(x)
    # g(x) = 3 + 2 * x - x ** 2 - 2 * x ** 3 + x ** 4
    # y(0) = y(1) = 1
    domain_ends = (0, 1)
    bcs = (
        (('dirichlet', 1), ('neumann', -1)),
    )

    def eqn1(x, y, y_x, y_xx):
        return torch.squeeze(3 + 2 * x - x ** 2 - 2 * x ** 3 + x ** 4 + y_xx[:, 0]) - y[:,0]**2
    
    # Each function in this list should return a 1D tensor (length = number of points in x)
    ODE_funcs = [eqn1]
    exact_sol = lambda x: 1 + x * (1 - x)
    
    no_epochs = 2000
    learning_rate = 0.005
elif BVP_NO == 7:
    # BVP proposed by Kathryn for Neumann BCs
    # ODE: -y'' + y^2 = g(x)
    # g(x) = 3 + 2 * x - x ** 2 - 2 * x ** 3 + x ** 4
    # y(0) = y(1) = 1
    alphas = (1, 0, -1)
    ns = (2, 1, 1)
    domain_ends = (0, 1)
    bcs = {'a':('neumann', 1),
            'b':('dirichlet', 1)}

    g_func = lambda x: 3 + 2*x - x**2 - 2*x**3 + x**4
    exact_sol = lambda x: 1 + x * (1 - x)
    
    no_epochs = 10000
    learning_rate = 0.004
elif BVP_NO == 8:
    # simple cos solution
    # neumann + dirichlet conditions
    domain_ends = (0, 2 * np.pi)
    bcs = (
        (('neumann', 0), ('dirichlet', 1)),
    )

    def eqn1(x, y, y_x, y_xx):
        return y[:,0] + torch.squeeze(y_xx[:,0])
    ODE_funcs = [eqn1]

    exact_sol = lambda x: np.cos(x) # UNIQUE soln

    no_epochs = 8000
    learning_rate = 0.001
elif BVP_NO == 9:
    alphas = (64, 0, 1)
    ns = (1, 1, 1)
    # Right domain end is about x = 2.55
    domain_ends = (0, 3 * np.pi / 8)
    bcs = {'a':('dirichlet', 1),
        'b':('neumann', 0)}
    g_func = lambda x: 0
    exact_sol = lambda x: np.cos(8 * x) # UNIQUE solution
    
    no_epochs = 30000
    learning_rate = 0.0008
elif BVP_NO == 10:
    # KATHRYN PROPOSED SYSTEM for u(x), v(x)
    # neumann + dirichlet conditions
    domain_ends = (0, 1)
    bcs = (
        (('dirichlet', 1), ('dirichlet', 1)),
        (('dirichlet', 0), ('dirichlet', 0)),
    )

    def eqn1(x, y, y_x, y_xx):
        return -y_xx[:,0] + torch.squeeze(x) * y[:,0] - y[:,1] - 2 - torch.squeeze(x)
    
    def eqn2(x, y, y_x, y_xx):
        return - torch.squeeze(y_xx[:,1]) + y[:,1] - 6 * torch.squeeze(x) + 2 - torch.squeeze(torch.squeeze(x) ** 2) * (1 - torch.squeeze(x))
    
    ODE_funcs = [eqn1,
                 eqn2]

    def exact_sol(x):
        return np.array([1 + x * (1 - x),
                         x**2 * (1 - x)])

    no_epochs = 600
    learning_rate = 0.03
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

    no_epochs = 10000
    learning_rate = 0.003

# Define BVP (routine)
my_bvp = BVP(
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
    loss_class = CustomLoss(bvp=my_bvp, gamma=8, bar_approach=BAR_APPROACH)
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
    loss_class = CustomLoss(bvp=my_bvp, gamma=10, bar_approach=BAR_APPROACH)
elif BVP_NO == 11:
    loss_class = CustomLoss(bvp=my_bvp, gamma=10, bar_approach=BAR_APPROACH)


# TRAINING POINTS
training_points = np.linspace(my_bvp.domain_ends[0], my_bvp.domain_ends[1], 50)
x_train = torch.tensor(training_points).reshape(len(training_points), 1).to(torch.float32).requires_grad_(True)

# MODEL
ANN_width = 5
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