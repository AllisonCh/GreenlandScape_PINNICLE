# Testing PINNICLE
# Infer basal friction coefficients using SSA

import pinnicle as pinn
import numpy as np
import deepxde as dde
import matplotlib.pyplot as plt
from datetime import datetime

# Set up some configurations
dde.config.set_default_float('float64')
dde.config.disable_xla_jit()
dde.config.set_random_seed(1234)


# Load the correct data

issm_filename = "Ryder_test_I20240729_140936"
datestr = datetime.now().strftime("%Y%m%d_%H%M%S")

# General parameters for training
# Setting up dictionaries
# order doesn't matter, but keys DO matter
hp = {}

# Load data ? 
# In data_size, each key:value pair defines a variable in the training. 
# if the key is not redefined in name_map, then it will be used as default 
# or set in the physics section above. The value associated with the key 
# gives the number of data points used for training.
# If the value is set to None, then only Dirichlet BC around the domain 
# boundary will be used for the corresponding key. If the variable is included
# in the training, but not given in data_size, then there will be no data for this variable in the training
issm = {}
issm["data_path"] = "~/PINNICLE/GreenlandScape_PINNICLE/Models/" + issm_filename + ".mat"
issm["data_size"] = {"u":8000, "v":8000, "s":8000, "H":None, "C":8000, "vel":8000}
hp["data"] = {"ISSM":issm}

hp["epochs"] = 100000
hp["learning_rate"] = 0.01
hp["loss_function"] = "MSE"
hp["save_path"] = "~/PINNICLE/GreenlandScape_PINNICLE/PINNs/" + issm_filename + "_P" + datestr
hp["is_save"] = True
hp["is_plot"] = True

# Set NN architecture
hp["activation"] = "tanh"
hp["initializer"] = "Glorot uniform"
hp["num_neurons"] = 20
hp["num_layers"] = 6
hp["input"] = ['y', 'x']

# Define domain of computation
hp["shapefile"] = "./Ryder_32_09.exp"
# Define number of collocation points used to evaluate PDE residual
hp["num_collocation_points"] = 2000

# Add physics
SSA = {}
SSA["scalar_variables"] = {"B":5.278336e+07}
hp["equations"] = {"SSA":SSA}
# MOLHO = {}
# MOLHO["scalar_variables"] = {"B":1.26802073401e+08}
# hp["equations"] = {"MOLHO":MOLHO}

# Add an additional loss function to balance the contributions between the fast flow and slow moving regions:
vel_loss = {}
vel_loss['name'] = "vel log"
vel_loss['function'] = "VEL_LOG"
vel_loss['weight'] = 1.0e-5
hp["additional_loss"] = {"vel":vel_loss}

experiment = pinn.PINN(hp)
experiment.update_parameters(hp)

# Now run the PINN model
experiment.compile()
# Train
experiment.train()

# Get data to save
pinn = experiment

import deepxde.backend as bkd

resolution = 200
    # generate 200x200 mesh on the domain
X, Y = np.meshgrid(np.linspace(pinn.params.nn.input_lb[0], pinn.params.nn.input_ub[0], resolution),
                   np.linspace(pinn.params.nn.input_lb[1], pinn.params.nn.input_ub[1], resolution))
X_nn = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))
grid_size = 2.0*(((pinn.params.nn.input_ub[0] - pinn.params.nn.input_lb[0])/resolution)**2+
                 ((pinn.params.nn.input_ub[1] - pinn.params.nn.input_lb[1])/resolution)**2)**0.5
if bkd.backend_name == "pytorch":
    grid_size = bkd.to_numpy(grid_size)

# predicted solutions
sol_pred = pinn.model.predict(X_nn)
plot_data = {k+"_pred":np.reshape(sol_pred[:,i:i+1], X.shape) for i,k in enumerate(pinn.params.nn.output_variables)}

import hdf5storage
import scipy
 
mat_data = {} # make a dictionary to store the MAT data in
vars2save = ['sol_pred','X_nn']
for i, var_curr in enumerate(vars2save):
    exec(f'mat_data[u"{var_curr}"] = {var_curr}')
 
hdf5storage.savemat(hp["save_path"] + 'predictions.mat', mat_data, format='7.3', oned_as='row', store_python_metadata=True)