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


issm_filename = "Ryder_test_I19-Dec-2024_3"
datestr = datetime.now().strftime("%d-%b-%y")

issm_pinn_path = issm_filename + "_P" + datestr + "_2"
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
flightTrack = {}
flightTrack["data_path"] = "./Ryder_xyz_ds.mat"
flightTrack["data_size"] = {"H": 20000}
flightTrack["name_map"] = {"H": "thickness"}
flightTrack["X_map"] = {"x": "x", "y":"y"}
flightTrack["source"] = "mat"
hp["data"] = {"ft": flightTrack}

issm = {}
issm["data_path"] = "./Models/" + issm_filename + ".mat"
issm["data_size"] = {"u":10000, "v":10000, "s":10000, "H":None, "C":10000, "B":10000}
hp["data"] = {"ISSM":issm, "ft":flightTrack} # hp = 'hyperparameters'

hp["epochs"] = int(1e6)
hp["learning_rate"] = 0.0005
hp["loss_function"] = "MSE"
hp["save_path"] = "./PINNs/" + issm_pinn_path
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
hp["num_collocation_points"] = 20000

# Add physics
yts = pinn.physics.Constants().yts
# SSA = {}
# SSA["scalar_variables"] = {"n":3} # -20 deg C
                    # u                     v                 s        H      C        B
# SSA["data_weights"] = [(1.0e-2*yts)**2.0, (1.0e-2*yts)**2.0, 5.0e-6, 2.0e-6, 7.0e-8, 1e-16]
# hp["equations"] = {"SSA":{"input":["x1", "x2"]}}
# hp["equations"] = {"SSA_VB":SSA}

MOLHO = {}
MOLHO["scalar_variables"] = {"B":2e+08}
hp["equations"] = {"MOLHO":MOLHO}
#                     #        u                 v                u_base               v_base            s        H      C
MOLHO["data_weights"] = [(1.0e-2*yts)**2.0, (1.0e-2*yts)**2.0, (1.0e-2*yts)**2.0, (1.0e-2*yts)**2.0, 1.0e-6, 1.0e-6, 1.0e-8]

hp['fft'] = True
hp['sigma'] = 5
hp['num_fourier_feature'] = 30

experiment = pinn.PINN(hp) # set up class PINN (in pinn.py in pinnicle package)
# experiment.update_parameters(hp)
# print(experiment.params) # make sure that settings are in correct spot (keys must be correct)

# Now run the PINN model
experiment.compile()

# Train
experiment.train()

# Show results
experiment.plot_predictions(X_ref=experiment.model_data.data["ISSM"].X_dict, sol_ref=experiment.model_data.data["ISSM"].data_dict)

# Save results 
import hdf5storage
import scipy

resolution = 150
    # generate 200x200 mesh on the domain
X, Y = np.meshgrid(np.linspace(experiment.params.nn.input_lb[0], experiment.params.nn.input_ub[0], resolution),
                   np.linspace(experiment.params.nn.input_lb[1], experiment.params.nn.input_ub[1], resolution))
X_nn = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))

# predicted solutions
sol_pred = experiment.model.predict(X_nn)
plot_data = {k+"_pred":np.reshape(sol_pred[:,i:i+1], X.shape) for i,k in enumerate(experiment.params.nn.output_variables)}

mat_data = {} # make a dictionary to store the MAT data in
vars2save = ['sol_pred','X_nn']
for i, var_curr in enumerate(vars2save):
    exec(f'mat_data[u"{var_curr}"] = {var_curr}')

hdf5storage.savemat(hp["save_path"] + '/' + issm_pinn_path + '_predictions.mat', mat_data, format='7.3', oned_as='row', store_python_metadata=True)
