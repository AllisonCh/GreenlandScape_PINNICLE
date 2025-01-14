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


issm_filename = "Ryder_issm2024-Dec-19_3"
datestr = datetime.now().strftime("%y-%b-%d")

issm_pinn_path = issm_filename + "_pinn" + datestr + "_2G"
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
wt_s = 5.0e-6
wt_H = 5.0e-6
wt_C = 1.0e-8
wt_B = 1e-16

# Load data ? 
flightTrack = {}
flightTrack["data_path"] = "./Ryder_xyz_ds.mat"
flightTrack["data_size"] = {"H": data_size_ft}
flightTrack["name_map"] = {"H": "thickness"}
flightTrack["X_map"] = {"x": "x", "y":"y"}
flightTrack["source"] = "mat"
hp["data"] = {"ft": flightTrack}

issm = {}
issm["data_path"] = "./Models/" + issm_filename + ".mat"
issm["data_size"] = {"u":data_size, "v":data_size, "s":data_size, "H":None, "C":None, "B":data_size}
hp["data"] = {"ISSM":issm, "ft":flightTrack} # hp = 'hyperparameters'

# Define number of collocation points used to evaluate PDE residual
hp["num_collocation_points"] = data_size*2

# Add physics
yts = pinn.physics.Constants().yts
SSA = {}
SSA["scalar_variables"] = {"n":3} # -20 deg C
                    # u        v      s    H      C     B
SSA["data_weights"] = [wt_uv, wt_uv, wt_s, wt_H, wt_C, wt_B]
# hp["equations"] = {"SSA":{"input":["x1", "x2"]}}
hp["equations"] = {"SSA_VB":SSA}

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

# Add an additional loss function to balance the contributions between the fast flow and slow moving regions:
# vel_loss = {}
# vel_loss['name'] = "vel log"
# vel_loss['function'] = "VEL_LOG"
# vel_loss['weight'] = 1.0e-5
# hp["additional_loss"] = {"vel":vel_loss}

experiment = pinn.PINN(hp) # set up class PINN (in pinn.py in pinnicle package)
# experiment.update_parameters(hp)
# print(experiment.params) # make sure that settings are in correct spot (keys must be correct)

# Now run the PINN model
experiment.compile()

# Train
experiment.train()
# Show results
# experiment.plot_predictions(X_ref=experiment.model_data.data["ISSM"].X_dict, sol_ref=experiment.model_data.data["ISSM"].data_dict)

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


# Prepare plotting data - load modules and define functions

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

def shadecalc_pre(s, u, v, resolution):
    # CALCULATE SURFACE SLOPE AND ALONG-FLOW SURFACE SLOPE AND MAKE FLOW-AWARE HILLSHADE
    print('Use filtered surface speed with elevation azimuth and calculate along-flow surface slope...')

    speed_filt = np.sqrt(np.square(u) + np.square(v))
    
    # surface-velocity and elevation-gradient flow azimuths
    az_speed = np.arctan2(v, u) # rad
    [elev_grad_y, elev_grad_x] = np.gradient(s, resolution)
    az_elev = np.arctan2(-elev_grad_y, -elev_grad_x) # rad
    
    [num_y, num_x]		= s.shape
    
    # extract trigonometric elements
    az_sin_cat, az_cos_cat = np.full((num_y, num_x, 2), np.nan), np.full((num_y, num_x, 2), np.nan)
    az_sin_cat[:, :, 0] = np.sin(az_speed)
    az_cos_cat[:, :, 0] = np.cos(az_speed)
    az_sin_cat[:, :, 1] = np.sin(az_elev)
    az_cos_cat[:, :, 1] = np.cos(az_elev)
    
    # weight filter exponentially using reference speed
    speed_az_decay = 100 # speed above which weight is unity for InSAR surface speeds, m/s (100 in GBTSv1)
    # speed_uncert_rel_decay = 0.25 # (0.1 in GBTSv1, 0.25 here), also fractional cutoff for speeds
    wt_az_elev = np.exp(-speed_filt / speed_az_decay) # + np.exp(-speed_uncert_rel_decay ./ (speed_uncert_filt ./ speed_filt)) % characteristic length of surface speed to unity ratio
    wt_az_elev[wt_az_elev > 1] = 1
    wt_az_speed = 1 - wt_az_elev
    wt_az_elev[np.isnan(az_speed)] = 1 # maximum weight (1) if the other is NaN
    wt_az_speed[np.isnan(az_elev)] = 1
    az_mean = np.arctan2((az_sin_cat[:, :, 0] * wt_az_speed) + (az_sin_cat[:, :, 1] * wt_az_elev), (az_cos_cat[:, :, 0] * wt_az_speed) + (az_cos_cat[:, :, 1] * wt_az_elev)) # mean azimuth, radians
    az_mean_cos = np.cos(az_mean) # rad
    az_mean_sin = np.sin(az_mean) # rad
    
    # Prepare to fix hillshade wrap issue
    az_wrap_cutoff = 150 # cutoff near 0/360 to address wrapping
    tmp1 = np.mod(np.degrees(az_mean), 360) # current projected Cartesian flow direction
    tmp1[(tmp1 >= az_wrap_cutoff) & (tmp1 <= 360 - az_wrap_cutoff)] = np.nan # add 360 deg to azimuths to get them well away from 0 (the wrapping problem)
    tmp2 = np.copy(tmp1) # interpolate cutoff azimuths onto full DEM grid (100 m)
    tmp3 = np.mod(np.degrees(az_mean), 360) # interpolate original azimuths onto full DEM grid
    tmp3[~np.isnan(tmp2)] = np.mod(tmp2[~np.isnan(tmp2)] - 360, 360) # take interpolated cutoff azimuths, remove the 360 deg, then wrap them, and put them where original azimuths were
    
    # use raw speed azimuth along margin where gridding breaks down
    az_speed = np.mod(np.degrees(np.arctan2(v, u)), 360)
    tmp3[np.isnan(tmp3) * ~np.isnan(az_speed)] = az_speed[np.isnan(tmp3) & ~np.isnan(az_speed)]
    az = np.radians(tmp3)
    return az


def shadecalc_alt(dem, resolution, az, el, zf):
    fy, fx = np.gradient(dem, resolution) # simple, unweighted gradient of immediate neighbours
    # Cartesian to polar coordinates for gradient:
    asp = np.arctan2(fy, fx)
    grad = np.hypot(fy, fx)
    # grad = np.radians(np.sqrt(fx**2 + fy**2))
    
    grad = np.arctan(grad * zf) # multiply gradient angle by z-factor
    hs = (np.cos(el) * np.cos(grad)) + (np.sin(el) * np.sin(grad) * np.cos(az - asp)) # ESRI's algorithm
    hs[hs < 0] = 0 # set hillshade values to min of 0
    return hs
# Prepare plotting data - SSA

grid_size = 501
# generate grid_size x grid_size mesh on the domain
X, Y = np.meshgrid(np.linspace(experiment.params.nn.input_lb[0], experiment.params.nn.input_ub[0], grid_size),
                   np.linspace(experiment.params.nn.input_lb[1], experiment.params.nn.input_ub[1], grid_size))
X_nn = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))

X_ref=experiment.model_data.data["ISSM"].X_dict
X_ref = np.hstack((X_ref['x'].flatten()[:,None],X_ref['y'].flatten()[:,None]))

resolution = X[0,1] - X[0,2]
yts = pinn.physics.Constants().yts

# reference data
sol_ref=experiment.model_data.data["ISSM"].data_dict
ref_data = {k:griddata(X_ref, sol_ref[k].flatten(), (X, Y), method='cubic') for k in experiment.params.nn.output_variables if k in sol_ref}

ref_data["u"] = yts*ref_data["u"]
ref_data["v"] = yts*ref_data["v"]
ref_data_plot = {"vel": np.sqrt(np.square(ref_data["u"]) + np.square(ref_data["v"])),
                 "hs":[], "C":ref_data["C"], "bed_elev":ref_data["s"] - ref_data["H"]}

# Get hillshade for reference surface
ref_az = shadecalc_pre(ref_data["s"], ref_data["u"], ref_data["v"], resolution)
#                                         dem         dx           az                 el             zf
ref_data_plot["hs"] = shadecalc_alt(ref_data["s"], resolution, ref_az + (np.pi/2), np.radians(30), 200) # hillshade lit across flow azimuth

ref_names = ref_data_plot.keys()

# predicted solutions
sol_pred = experiment.model.predict(X_nn)
pred_data = {k:np.reshape(sol_pred[:,i:i+1], X.shape) for i,k in enumerate(experiment.params.nn.output_variables)}
data_names = pred_data.keys()

pred_data["u"] = yts*pred_data["u"]
pred_data["v"] = yts*pred_data["v"]
pred_data["vel"] = np.sqrt(np.square(pred_data["u"]) + np.square(pred_data["v"]))
pred_data["bed_elev"] = ref_data["s"] - pred_data["H"]

# Get hillshade for predicted surface
pred_az = shadecalc_pre(pred_data["s"], pred_data["u"], pred_data["v"], resolution)
#                                         dem         dx           az                 el             zf
pred_data["hs"] = shadecalc_alt(pred_data["s"], resolution, pred_az + (np.pi / 2), np.radians(30), 200) # hillshade lit across flow azimuth

# Get percent differences
perc_diff = {}
perc_diff["vel"]= ((pred_data["vel"] - ref_data_plot["vel"])/ref_data_plot["vel"])*100
perc_diff["s"]= ((pred_data["s"] - ref_data["s"])/ref_data["s"])*100
perc_diff["C"]= ((pred_data["C"] - ref_data_plot["C"])/ref_data_plot["C"])*100
perc_diff["bed_elev"]= ((pred_data["bed_elev"] - ref_data_plot["bed_elev"])/ref_data_plot["bed_elev"])*100


# Get colorbar ranges for plotting data
cranges = {name:[np.round(np.min(ref_data_plot[name]),decimals=-1), np.round(np.max(ref_data_plot[name]),decimals=-1)] for name in ref_names}
cranges["hs"] = [0.0, 1.0]

q75_pd = {name:np.quantile(np.abs(perc_diff[name]),0.75) for name in perc_diff.keys()}
cranges_pd = {name:[np.round(-1*q75_pd[name]), np.round(q75_pd[name])] for name in perc_diff.keys()}


clabels = {name:[] for name in ref_names}
clabels["vel"] = "(m/yr)"
clabels["hs"] = ""
clabels["C"] = "($Pa^{1/2} m^{-1/6} s^{1/6}$)"
clabels["bed_elev"] = "(m)"

cmaps = {name:[] for name in ref_names}
cmaps["vel"] = plt.get_cmap("magma",10)
cmaps["hs"] = plt.get_cmap("gray", 100)
cmaps["C"] = plt.get_cmap("cividis", 10)
cmaps["bed_elev"] = plt.get_cmap("terrain", 10)

cols = len(ref_names)

# Make the plots
import string

titles = {"vel":"Velocity", "hs":"Surface hillshade", "C":"Basal friction\ncoefficient",
          "bed_elev":"Bed elevation\n(GrIMP - H)"}
extends = {"vel":"max", "hs":"neither","C":"both","bed_elev":"both"}


alphabet = list(string.ascii_lowercase)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 12

# for ax, name in zip(axs.ravel(), ref_data.keys()):
fig, axs = plt.subplots(3, cols, figsize=(12,8),dpi=300)
fig.subplots_adjust(left=0.1, hspace=0.1, wspace=0.45)
for ax, name in zip(axs[0], ref_data_plot.keys()):
    vr = cranges.setdefault(name, [None, None])
    im = ax.imshow(ref_data_plot[name], interpolation='nearest', cmap=cmaps[name],
                   extent=[X.min(), X.max(), Y.min(), Y.max()],
                   vmin=vr[0], vmax=vr[1],
                   aspect='equal', origin='lower')
    ax.set_title(titles[name])
    ax.tick_params(left = False, right = False , labelleft = False ,
                   labelbottom = False, bottom = False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.05, location="right", pad=0.02, extend = extends[name], ticks=vr) 
    

for ax, name in zip(axs[1], ref_data_plot.keys()):
    vr = cranges.setdefault(name, [None, None])
    im = ax.imshow(pred_data[name], interpolation='nearest', cmap=cmaps[name],
                   extent=[X.min(), X.max(), Y.min(), Y.max()],
                   vmin=vr[0], vmax=vr[1],
                   origin='lower', aspect='equal')
    # ax.set_title(name+"_pred")
    ax.tick_params(left = False, right = False , labelleft = False , 
                   labelbottom = False, bottom = False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, location="right", extend = extends[name], ticks=vr) 
    cbar.ax.set_title(clabels[name])

# fig, axs = plt.subplots(math.floor(n/cols), cols, figsize=(12,9))
for ax, name in zip(axs[2], perc_diff.keys()):
    vr = cranges_pd.setdefault(name, [None, None])
    im = ax.imshow(perc_diff[name], interpolation='nearest', cmap=plt.get_cmap('RdBu_r',11),
                   extent=[X.min(), X.max(), Y.min(), Y.max()],
                   vmin=vr[0], vmax=vr[1],
                   aspect='equal', origin='lower')
    ax.tick_params(left = False, right = False , labelleft = False ,
                   labelbottom = False, bottom = False)
    cbar = fig.colorbar(im, ax=ax, fraction = 0.05, location="right", extend="both",ticks=vr) 
    cbar.ax.set_title("(%)") #, y=1.15, rotation=0)

axs[0,0].set_ylabel("Observations/\nmodel reference")
axs[1,0].set_ylabel("Prediction")
axs[2,0].set_ylabel("Relative difference")

for ax, label in zip(axs.ravel(), alphabet):
    ax.annotate("("+label+")",
    xy=(0, 1), xycoords='axes fraction',
    xytext=(+0.5, -0.5), textcoords='offset fontsize',
    fontsize='small', verticalalignment='top', fontfamily='serif',
    bbox=dict(facecolor='1.0', edgecolor='none', pad=2.0,alpha=0.7))

# plt.savefig(hp["save_path"]+"/2Dsolutions")

plt.savefig(experiment.params.param_dict["save_path"]+"/2Dsolutions")

# Plot history on one axis
from pinnicle.utils.history import load_dict_from_json
experiment.history = load_dict_from_json(model_path,"history.json")

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 8

loss_keys = [k for k in experiment.history.keys() if k != "steps"]

fig = plt.figure(dpi = 300) 
plt.plot(experiment.history['fSSA_VB1'], "k-", label="SSA PDE 1", linewidth=2)
plt.plot(experiment.history['fSSA_VB2'], label="SSA PDE 2", linestyle="-", c='0.4', linewidth=2)
plt.plot(experiment.history['u'], label='Surface velocity (x-component)', linewidth=4, c=[0.85,0.33,0.10])
plt.plot(experiment.history['v'], label='Surface velocity (y-component)', linewidth=4, c=[0.69,0.16,0.69])
plt.plot(experiment.history['s'], label='Surface elevation', linewidth=6, c=[0.58,0.39,0.39])
plt.plot(experiment.history['H'], label='Ice thickness', linewidth=4, c='g', linestyle=":")
plt.plot(experiment.history['C'], label='Basal friction coefficient', linewidth=4, linestyle="-.", c=[0.30,0.75,0.93])

plt.legend(loc=1,fontsize='small')
plt.yscale('log')
plt.xlabel('Steps (*10^4)')
plt.ylabel('Loss function value')

plt.savefig(experiment.params.param_dict["save_path"]+"/History_1ax")