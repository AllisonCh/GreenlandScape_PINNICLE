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

# Start loading data
issm_filename = "Ryder_issm2024-Dec-19_3"
datestr = datetime.now().strftime("%y-%b-%d")

issm_pinn_path = issm_filename + "_pinn" + datestr + "_1G"
# General parameters for training
# Setting up dictionaries: order doesn't matter, but keys DO matter
hp = {}
# Define domain of computation
hp["shapefile"] = "./Ryder_32_09.exp"
# Define hyperparameters
hp["epochs"] = int(1e5)
hp["learning_rate"] = 0.001
hp["loss_function"] = "MSE"

yts = pinn.physics.Constants().yts
data_size = 8000
data_size_ft = 10000
wt_uv = (1.0e-2*yts)**2.0
wt_uvb = (1.0e-2*yts)**2.0
wt_s = 5.0e-6
wt_H = 1.0e-5
wt_C = 1.0e-8

# Load data
flightTrack = {}
flightTrack["data_path"] = "Ryder_xyz_ds.mat"
flightTrack["data_size"] = {"H": data_size_ft}
flightTrack["X_map"] = {"x": "x", "y":"y"}
flightTrack["name_map"] = {"H": "thickness"}
flightTrack["source"] = "mat"

velbase = {}
velbase["data_path"] = "./Ryder_vel_base.mat"
velbase["data_size"] = {"u_base":data_size, "v_base":data_size}
velbase["name_map"] = {"u_base":"md_u_base", "v_base":"md_v_base"}
velbase["X_map"] = {"x":"x", "y":"y"}
velbase["source"] = "mat"

issm = {}
issm["data_path"] = "./Models/" + issm_filename + ".mat"
issm["data_size"] = {"u":data_size, "v":data_size, "s":data_size, "H":None, "C":None}
hp["data"] = {"ISSM":issm, "ft":flightTrack,"velbase":velbase} # hp = 'hyperparameters'

# Define number of collocation points used to evaluate PDE residual
hp["num_collocation_points"] = data_size*2

# Add physics
MOLHO = {}
MOLHO["scalar_variables"] = {"B":2e+08}
hp["equations"] = {"MOLHO":MOLHO}
#                       # u     v       u_base  v_base  s     H      C
MOLHO["data_weights"] = [wt_uv, wt_uv, wt_uvb, wt_uvb, wt_s, wt_H, wt_C]

MOLHO["output_lb"] =    [-1.0e4/yts,         -1.0e4/yts,         -1.0e2/yts,         -1.0e2/yts,     -1.0e3,  10.0, 0.01]
MOLHO["output_ub"] =    [1.0e4/yts,           1.0e4/yts,          1.0e2/yts,          1.0e2/yts,      4.0e3,  4.0e3, 1.0e4]
MOLHO["variable_lb"] =  [-1.0e4/yts,         -1.0e4/yts,         -1.0e2/yts,         -1.0e2/yts,     -1.0e3,  10.0, 0.01]
MOLHO["variable_ub"] =  [1.0e4/yts,           1.0e4/yts,          1.0e2/yts,          1.0e2/yts,      4.0e3,  4.0e3, 1.0e4]

# Set NN architecture
hp["activation"] = "tanh"
hp["initializer"] = "Glorot uniform"
hp["num_neurons"] = 20
hp["num_layers"] = 6
hp["input"] = ['y', 'x']

hp['fft'] = True
hp['sigma'] = 5
hp['num_fourier_feature'] = 30

hp["save_path"] = "./PINNs/" + issm_pinn_path
hp["is_save"] = True
hp["is_plot"] = True

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
# plot_data = {k+"_pred":np.reshape(sol_pred[:,i:i+1], X.shape) for i,k in enumerate(experiment.params.nn.output_variables)}

mat_data = {} # make a dictionary to store the MAT data in
vars2save = ['sol_pred','X_nn']
for i, var_curr in enumerate(vars2save):
    exec(f'mat_data[u"{var_curr}"] = {var_curr}')

hdf5storage.savemat(hp["save_path"] + '/' + issm_pinn_path + '_predictions.mat', mat_data, format='7.3', oned_as='row', store_python_metadata=True)


# Plotting attempts

import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from scipy.interpolate import griddata
from scipy.spatial import cKDTree as KDTree
import scipy.io as sio
import pandas as pd
from scipy.stats import iqr


X_ref=experiment.model_data.data["ISSM"].X_dict
X_ref = np.hstack((X_ref['x'].flatten()[:,None],X_ref['y'].flatten()[:,None]))

sol_ref=experiment.model_data.data["ISSM"].data_dict
ref_data = {k:griddata(X_ref, sol_ref[k].flatten(), (X, Y), method='cubic') for k in experiment.params.nn.output_variables if k in sol_ref}

pred_data = {k:np.reshape(sol_pred[:,i:i+1], X.shape) for i,k in enumerate(experiment.params.nn.output_variables)}
vranges = {k+"_pred":[experiment.params.nn.output_lb[i], experiment.params.nn.output_ub[i]] for i,k in enumerate(experiment.params.nn.output_variables)}

ref_names = ref_data.keys()
data_names = pred_data.keys()

ref_data["u"] = yts*ref_data["u"]
ref_data["v"] = yts*ref_data["v"]

pred_data["u"] = yts*pred_data["u"]
pred_data["v"] = yts*pred_data["v"]
pred_data["u_base"] = yts*pred_data["u_base"]
pred_data["v_base"] = yts*pred_data["v_base"]

cranges = {name:[np.round(np.min(ref_data[name]),decimals=-1), np.round(np.max(ref_data[name]),decimals=-1)] for name in ref_names}
cranges["u_base"] = cranges["u"]
cranges["v_base"] = cranges["v"]

clabels = {name:[] for name in ref_names}
clabels["u"] = "m/yr"
clabels["v"] = "m/yr"
clabels["s"] = "m"
clabels["C"] = "Pa^1/2 m^-1/6 s^1/6"
clabels["H"] = "m"

cmaps = {name:[] for name in ref_names}
cmaps["u"] = plt.get_cmap("magma", 10)
cmaps["v"] = plt.get_cmap("magma", 10)
cmaps["s"] = plt.get_cmap("gist_earth", 10)
cmaps["C"] = plt.get_cmap("cividis", 10)
cmaps["H"] = plt.get_cmap("ocean", 10)

perc_diff = {name:((pred_data[name] - ref_data[name])/ref_data[name])*100 for name in ref_names}
q75_pd = {name:np.quantile(np.abs(perc_diff[name]),0.75) for name in ref_names}
cranges_pd = {name:[-1*q75_pd[name], q75_pd[name]] for name in ref_names}
# cranges_pd = {name:[-np.round(np.max(np.abs(perc_diff[name])), decimals=-1), np.round(np.max(np.abs(perc_diff[name])), decimals=-1)] for name in ref_names}

n = len(ref_data)
# if cols is None:
cols = len(ref_names)

# fig, axs = plt.subplots(math.ceil(n/cols), cols, figsize=(12,9))
# for ax, name in zip(axs.ravel(), ref_data.keys()):
fig, axs = plt.subplots(3, cols, figsize=(12,9))
for ax, name in zip(axs[0], ref_data.keys()):
    vr = cranges.setdefault(name, [None, None])
    im = ax.imshow(pred_data[name], interpolation='nearest', cmap=cmaps[name],
                   extent=[X.min(), X.max(), Y.min(), Y.max()],
                   vmin=vr[0], vmax=vr[1],
                   origin='lower', aspect='equal')
    ax.set_title(name+"_pred")
    ax.tick_params(left = False, right = False , labelleft = False , 
                   labelbottom = False, bottom = False)
    fig.colorbar(im, ax=ax, location="bottom")

# fig, axs = plt.subplots(math.floor(n/cols), cols, figsize=(12,9))
for ax, name in zip(axs[1], ref_data.keys()):
    vr = cranges.setdefault(name, [None, None])
    im = ax.imshow(ref_data[name], interpolation='nearest', cmap=cmaps[name],
                   extent=[X.min(), X.max(), Y.min(), Y.max()],
                   vmin=vr[0], vmax=vr[1],
                   aspect='equal', origin='lower')
    ax.set_title(name+"_ref")
    ax.tick_params(left = False, right = False , labelleft = False ,
                   labelbottom = False, bottom = False)
    fig.colorbar(im, ax=ax, label=clabels[name], fraction=0.1, orientation="horizontal", location="bottom") 
    
# fig, axs = plt.subplots(math.floor(n/cols), cols, figsize=(12,9))
for ax, name in zip(axs[2], ref_data.keys()):
    vr = cranges_pd.setdefault(name, [None, None])
    im = ax.imshow(perc_diff[name], interpolation='nearest', cmap=plt.get_cmap('RdBu_r',11),
                   extent=[X.min(), X.max(), Y.min(), Y.max()],
                   vmin=vr[0], vmax=vr[1],
                   aspect='equal', origin='lower')
    ax.tick_params(left = False, right = False , labelleft = False ,
                   labelbottom = False, bottom = False)
    fig.colorbar(im, ax=ax, orientation="horizontal", location="bottom") 

plt.savefig(hp["save_path"]+"/2Dsolutions")
