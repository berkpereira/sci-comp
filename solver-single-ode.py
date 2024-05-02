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
FIGSIZE = (10, 6)
torch.manual_seed(42)
####################################################################################################
####################################################################################################
####################################################################################################
class BVP:
    def __init__(self, alphas, ns, g_func, domain_ends, bcs):
        """
        Initializes a boundary value problem for a second-order ODE.
        
        Args:
            alphas (tuple): A 3-tuple of coefficients (alpha0, alpha1, alpha2) for the ODE terms.
            ns (tuple): A 3-tuple of powers (n0, n1, n2) for the ODE terms.
            g_func (callable): The right-hand side function g(x).
            domain_ends (tuple): The domain ends (a, b).
            bcs (dictionary): Separable boundary conditions y(a) and y(b).
        """
        # Unpack the tuples
        self.alpha0, self.alpha1, self.alpha2 = alphas
        self.n0, self.n1, self.n2 = ns
        
        self.g_func = g_func
        self.domain_ends = domain_ends
        self.bcs = bcs
    def eval_ode(self, x, y, y_x, y_xx):
        """
        Evaluates the left-hand side of the ODE given y, y', and y''.
        Given a correct solution, THIS SHOULD EQUAL ZERO
        """
        term2 = self.alpha2 * (y_xx ** self.n2)
        term1 = self.alpha1 * (y_x ** self.n1)
        term0 = self.alpha0 * (y ** self.n0)
        return term2 + term1 + term0 - self.g_func(x)

class BVPflexible:
    def __init__(self, ODE_func, domain_ends, bcs):
        """
        Initializes a boundary value problem for a single second-order ODEs.

        Args:
            terms_funcs (list of callables): Each item corresponds to an ODE and contains a callables that returns the residual.
            domain_ends (tuple): The domain ends (a, b).
            bcs (dictionary): Boundary conditions with details for each end ('a' and 'b') and possibly for each variable.
        """
        
        # The bcs dictionary's form has a lot of info about the BVP
        self.ODE_func = ODE_func
        self.domain_ends = domain_ends
        self.bcs = bcs

    # y-related inputs should be vectors as appropriate in ODE system cases
    def eval_ode(self, x, y, y_x, y_xx):
        """
        Evaluates residual.

        Args:
            x (torch.Tensor): Input tensor of independent variable values.
            y (torch.Tensor): Tensor of shape (num_points) of function values at x.
            y_x (torch.Tensor): Tensor of shape (num_points) of first derivatives at x.
            y_xx (torch.Tensor): Tensor of shape (num_points) of second derivatives at x.
        
        Returns:
            torch.Tensor: A tensor representing the residuals for the ODE.
        """
        
        residuals = self.ODE_func(x, y, y_x, y_xx)
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
        y_a_type, y_b_type = bcs['a'][0], bcs['b'][0]
        y_a, y_b = bcs['a'][1], bcs['b'][1]
        if y_a_type == 'dirichlet' and y_b_type == 'dirichlet':
            # 1st scaling option (seems to work well)
            y_bar = (x - a) * (b - x) * y_hat + (x - a)/(b - a) * y_b + (b - x) / (b - a) * y_a
            
            # 2nd scaling option (seems to work worse)
            # y_bar = y_hat + (b - x)*(y_a - y_hat[0])/(b - a) + (x - a)*(y_b - y_hat[-1])/(b - a)
        elif y_a_type == 'dirichlet' and y_b_type == 'neumann':
            y_hat_b = self.stack(b)
            # evaluate derivative at x = b
            y_hat_x_b = torch.autograd.grad(y_hat_b, b, torch.ones_like(y_hat_b), create_graph=True)[0]
            
            # 1st scaling approach (Kathryn-proposed) (I think I copied down wrong, does not enforce BC)
            # y_bar = y_b * x + y_a - a * y_b + (x - a) * (y_hat - y_hat_b - y_hat_x_b) / (b - a)
            # MY FIX to Kathryn proposal
            y_bar = y_b * x + y_a - a * y_b + (x - a) * ((y_hat - y_hat_b)/(b - a) - y_hat_x_b)
            
            # 2nd scaling approach (Maria-proposed)
            # y_bar = y_hat + (y_a - y_hat[0]) + (x - a) * (y_b - y_hat_x_b)
        elif y_a_type == 'neumann' and y_b_type == 'dirichlet':
            y_hat_a = self.stack(a)
            # evaluate derivative at x = a
            y_hat_x_a = torch.autograd.grad(y_hat_a, a, torch.ones_like(y_hat_a), create_graph=True)[0]
            
            # 1st scaling approach (Kathryn-proposed) (I think I copied down wrong, does not enforce BC)
            # y_bar = y_a * x + y_b - b * y_a + (x - b) * (y_hat - y_hat_a - y_hat_x_a) / (b - a)
            # MY FIX to Kathryn proposal
            y_bar = y_a * x + y_b - b * y_a + (b - x) * ((y_hat - y_hat_a)/(b - a) - y_hat_x_a)
            # 2nd scaling approach (Maria-proposed)
            # y_bar = y_hat + (y_b - y_hat[-1]) + (x - b) * (y_a - y_hat_x_a)
        elif y_a_type == 'neumann' and y_b_type == 'neumann':
            pass # Still have to implement 
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
        y_x = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        y_xx = torch.autograd.grad(y_x, x, torch.ones_like(y_x), create_graph=True)[0]
        
        ode_loss = torch.mean(self.bvp.eval_ode(x, y, y_x, y_xx) ** 2)
        if self.bar_approach:
            return ode_loss
        else:
            y_a, y_b = bcs['a'][1], bcs['b'][1]
            bc_loss = self.gamma * ((y[0] - y_a) ** 2 + (y[-1] - y_b) ** 2)
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
    y_pred_numpy = y_pred_tensor.detach().numpy().flatten()
    # Plot predictions
    plt.figure(figsize=FIGSIZE)
    plt.plot(x_train_numpy, y_pred_numpy, label='NN Predictions', color='r', linestyle='--', marker='o')
    title_str = 'NN Predictions'
    
    # Analytical solution if provided
    if exact_sol_func is not None:
        y_exact_numpy = exact_sol_func(x_train_numpy)
        plt.plot(x_train_numpy, y_exact_numpy, label='Analytical Solution', color='b', linestyle='-')
        title_str = 'NN Predictions vs Exact Solution'
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title_str)
    plt.legend()
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
BVP_NO = 1
BAR_APPROACH = True
if BVP_NO == 0:
    # BVP proposed by Kathryn
    # ODE: -y'' + y^2 = g(x)
    # g(x) = 3 + 2 * x - x ** 2 - 2 * x ** 3 + x ** 4
    # y(0) = y(1) = 1
    domain_ends = (0, 1)
    bcs = {'a':('dirichlet', 1),
            'b':('dirichlet', 1)}
    def ODE_func(x, y, y_x, y_xx):
        return y_xx - y**2 + 3 + 2*x - x**2 - 2*x**3 + x**4

    exact_sol = lambda x: 1 + x * (1 - x)
    
    no_epochs = 300
    learning_rate = 0.004
elif BVP_NO == 1:
    # BVP with boundary layer solution
    eps = 0.02
    
    def ODE_func(x, y, y_x, y_xx):
        return y_x - eps * y_xx - 1
    
    domain_ends = (0, 1)
    bcs = {'a':('dirichlet', 1),
           'b':('dirichlet', 1)}
    exact_sol = None
    
    no_epochs = 12000
    learning_rate = 0.0075
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
    alphas = (1, 0, 1)
    ns = (1, 0, 1)
    domain_ends = (0, 3 * np.pi / 2)
    bcs = (0, 1)
    g_func = lambda x: 0
    exact_sol = lambda x: - np.sin(x)
    no_epochs = 5000
    learning_rate = 0.002
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
    alphas = (1, 0, -1)
    ns = (2, 1, 1)
    domain_ends = (0, 1)
    bcs = {'a':('dirichlet', 1),
            'b':('neumann', -1)}
    g_func = lambda x: 3 + 2*x - x**2 - 2*x**3 + x**4
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
    alphas = (1, 0, 1)
    ns = (1, 0, 1)
    domain_ends = (0, 2 * np.pi)
    bcs = {'a':('neumann', 0),
        'b':('dirichlet', 1)}
    g_func = lambda x: 0
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

# Define BVP (routine)
my_bvp = BVPflexible(ODE_func, domain_ends, bcs)

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

# TRAINING POINTS
training_points = np.linspace(my_bvp.domain_ends[0], my_bvp.domain_ends[1], 50)
x_train = torch.tensor(training_points).reshape(len(training_points), 1)
x_train = x_train.to(torch.float32).requires_grad_(True)
# MODEL
ANN_width = 50
ANN_depth = 1
model = NeuralNetwork(my_bvp, 1, 1, ANN_width, ANN_depth, bar_approach=BAR_APPROACH)
# OPTIMISER
optimiser = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
# Loss
loss_values = train_model(model, optimiser, my_bvp, loss_class, x_train, no_epochs)
# PLOTTING
plot_predictions(model, x_train, exact_sol_func=exact_sol)
plot_loss_vs_epoch(loss_values)
plot_ode_residuals(model, my_bvp, x_train)