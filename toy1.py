import pandas as pd
import numpy as np
import esda
# import pysal
import libpysal
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from IPython.display import display, clear_output
import warnings

warnings.simplefilter("ignore")

import sys
sys.path.append('src')
from spacegan_method import SpaceGAN
from spacegan_selection import get_spacegan_config, compute_metrics
from spacegan_utils import gaussian, rmse, mad, pearsoncorr, mie, moranps, mase_1, mape, smape, eool, msis_1, get_neighbours_featurize
from spacegan_config import Generator, Discriminator
# %matplotlib inline

# dataset
df = pd.read_csv("data/toy1.csv")
coord_vars = ["longitude", "latitude"] #Define spatial coordinates
cond_vars = ["z"] + coord_vars #Define predictors
cont_vars = ["z","y"] + coord_vars #Define which neighbour features to use as context variables
output_vars = ["y"]
neighbours = 8 #Define the number of neihgbours to use

# plotting
#if ax1.collections: #Delete plot if it exists already
#  ax1.clear()
  
fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
gen_seq = df[["y"]].values.astype(float)
norm_gan_mean = (gen_seq - min(gen_seq)) / (max(gen_seq) - min(gen_seq))
colors = cm.rainbow(norm_gan_mean)

# plotting
for lat, long, c in zip(df["latitude"], df["longitude"], colors):
  ax1.scatter(lat, long, color=c)
  
ax1.set_xlabel(r'$c^{(1)}$', fontsize=14)
ax1.set_ylabel(r'$c^{(2)}$', fontsize=14)
ax1.set_title("Observed")


# problem configuration
prob_config = {"epochs": 20000,
               "batch_size": 100,
               "device": torch.device("cuda"),
               "cond_dim": len(cond_vars) + (neighbours * len(cont_vars)),  # conditional information size
               "output_dim": len(output_vars),  # size of output
               "noise_dim": len(cond_vars) + (neighbours * len(cont_vars)),  # size of noise
               "noise_type": gaussian,  # type of noise and dimension used
               "noise_params": None,  # other params for noise (loc, scale, etc.) pass as a dict
               "scale_x": StandardScaler(),  # a sklearn.preprocessing scaling method
               "scale_y": StandardScaler(),  # a sklearn.preprocessing scaling method
               "print_results": False
               }

# additional Generator params
prob_config["gen_opt"] = torch.optim.SGD
prob_config["gen_opt_params"] = {"lr": 0.01}

# additional Discriminator params
prob_config["disc_opt"] = torch.optim.SGD
prob_config["disc_opt_params"] = {"lr": 0.01}

# loss function
prob_config["adversarial_loss"] = torch.nn.BCELoss()

# checkpointing configuration
check_config = {
    "check_interval": 100,  # for model checkpointing
    "generate_image": False,
    "n_samples": 20,
    "perf_metrics": {"RMSE": rmse,
                     "MIE": mie,
                     },
    "pf_metrics_setting": {
        "RMSE": {"metric_level": "agg_metrics",
             "rank_function": np.argmin,
             "agg_function": lambda x: np.array(x)
             },
        "MIE": {"metric_level": "agg_metrics",
                "rank_function": np.argmin,
                "agg_function": lambda x: np.array(x)
               },
    },
    "agg_funcs": {"avg": np.mean,
                  "std": np.std
                 },
    "sample_metrics": False,
    "agg_metrics": True
}

# train the model

# neighbours
df, neighbour_list = get_neighbours_featurize(df, coord_vars, cont_vars, neighbours)

# data structures
target = df[output_vars].values
cond_input = df[cond_vars + neighbour_list].values
coord_input = df[coord_vars].values
prob_config["output_labels"] = output_vars
prob_config["input_labels"] = cond_vars + neighbour_list

# pre-instantiation
disc_method = Discriminator(prob_config["output_dim"], prob_config["cond_dim"])
disc_method.to(prob_config["device"])
gen_method = Generator(prob_config["cond_dim"], prob_config["noise_dim"], prob_config["output_dim"])
gen_method.to(prob_config["device"])

# training SpaceGAN
spacegan = SpaceGAN(prob_config, check_config, disc_method, gen_method)
spacegan.train(x_train=cond_input, y_train=target, coords=coord_input)

# export final model and data
spacegan.checkpoint_model(spacegan.epochs) 
spacegan.df_losses.to_pickle("grid_spaceganlosses.pkl.gz")


'''
# pick the best Generator (G) as determined by the MIE and the RMSE criterion.

# computing metrics
gan_metrics = compute_metrics(target, cond_input, prob_config, check_config, coord_input, neighbours)

# selecting and sampling gan
for criteria in list(check_config["perf_metrics"].keys()):
    # find best config
    criteria_info = check_config["pf_metrics_setting"][criteria]
    perf_metrics = gan_metrics[criteria_info["metric_level"]]
    perf_values = criteria_info["agg_function"](perf_metrics[[criteria]])
    best_config = perf_metrics.index[criteria_info["rank_function"](perf_values)]

    # get and set best space gan
    best_spacegan = get_spacegan_config(int(best_config), prob_config, check_config, cond_input, target)
    # training samples
    gan_samples_df = pd.DataFrame(index=range(cond_input.shape[0]), columns=cond_vars + neighbour_list + output_vars)
    gan_samples_df[cond_vars + neighbour_list] = cond_input
    gan_samples_df[output_vars] = target
    for i in range(check_config["n_samples"]):
        gan_samples_df["sample_" + str(i)] = best_spacegan.predict(gan_samples_df[cond_vars + neighbour_list])

    # export results
    gan_samples_df.to_pickle("grid_" + criteria + ".pkl.gz")
gan_metrics["agg_metrics"].to_pickle("grid_checkmetrics.pkl.gz")



# plot the results!

# show highlights
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
gan_metrics["agg_metrics"].plot(ax=ax1)

# generate chart
gen_seq = gan_samples_df[["sample_" + str(x) for x in range(20)]].mean(axis=1)
norm_gan_mean = (gen_seq - min(gen_seq)) / (max(gen_seq) - min(gen_seq))
colors = cm.rainbow(norm_gan_mean)

# plotting
for lat, long, c in zip(df["latitude"], df["longitude"], colors):
    ax2.scatter(lat, long, color=c)
ax2.set_xlabel(r'$c^{(1)}$', fontsize=14)
ax2.set_ylabel(r'$c^{(2)}$', fontsize=14)
ax2.set_title("SpaceGAN - Best " + criteria)



# plot the best generator after RMSE selection

#load rmse selection results
gan_samples_df = pd.read_pickle("./grid_RMSE.pkl.gz")

# show highlights
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
gan_metrics["agg_metrics"].plot(ax=ax1)

# generate chart
gen_seq = gan_samples_df[["sample_" + str(x) for x in range(20)]].mean(axis=1)
norm_gan_mean = (gen_seq - min(gen_seq)) / (max(gen_seq) - min(gen_seq))
colors = cm.rainbow(norm_gan_mean)

# plotting
for lat, long, c in zip(df["latitude"], df["longitude"], colors):
    ax2.scatter(lat, long, color=c)
ax2.set_xlabel(r'$c^{(1)}$', fontsize=14)
ax2.set_ylabel(r'$c^{(2)}$', fontsize=14)
ax2.set_title("SpaceGAN - Best RMSE")



# selection

iteration = 3000

# get and set best space gan
iter_spacegan = get_spacegan_config(iteration, prob_config, check_config, cond_input, target)

# training samples
gan_samples_df = pd.DataFrame(index=range(cond_input.shape[0]), columns=cond_vars + neighbour_list + output_vars)
gan_samples_df[cond_vars + neighbour_list] = cond_input
gan_samples_df[output_vars] = target
for i in range(check_config["n_samples"]):
    gan_samples_df["sample_" + str(i)] = iter_spacegan.predict(gan_samples_df[cond_vars + neighbour_list])
    
# generate chart
fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
gen_seq = gan_samples_df[["sample_" + str(x) for x in range(1)]].mean(axis=1)
norm_gan_mean = (gen_seq - min(gen_seq)) / (max(gen_seq) - min(gen_seq))
colors = cm.rainbow(norm_gan_mean)

# plotting
for lat, long, c in zip(df["latitude"], df["longitude"], colors):
    ax1.scatter(lat, long, color=c)
ax1.set_xlabel(r'$c^{(1)}$', fontsize=14)
ax1.set_ylabel(r'$c^{(2)}$', fontsize=14)
ax1.set_title("SpaceGAN (RMSE) - Iteration " + str(iteration))




iteration = 3000

# get and set best space gan
iter_spacegan = get_spacegan_config(iteration, prob_config, check_config, cond_input, target)

#load mie selection results
gan_samples_df = pd.read_pickle("./grid_MIE.pkl.gz")

# training samples
gan_samples_df = pd.DataFrame(index=range(cond_input.shape[0]), columns=cond_vars + neighbour_list + output_vars)
gan_samples_df[cond_vars + neighbour_list] = cond_input
gan_samples_df[output_vars] = target
for i in range(check_config["n_samples"]):
    gan_samples_df["sample_" + str(i)] = iter_spacegan.predict(gan_samples_df[cond_vars + neighbour_list])
    
# generate chart
fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
gen_seq = gan_samples_df[["sample_" + str(x) for x in range(1)]].mean(axis=1)
norm_gan_mean = (gen_seq - min(gen_seq)) / (max(gen_seq) - min(gen_seq))
colors = cm.rainbow(norm_gan_mean)

# plotting
for lat, long, c in zip(df["latitude"], df["longitude"], colors):
    ax1.scatter(lat, long, color=c)
ax1.set_xlabel(r'$c^{(1)}$', fontsize=14)
ax1.set_ylabel(r'$c^{(2)}$', fontsize=14)
ax1.set_title("SpaceGAN (MIE) - Iteration " + str(iteration))




#Load loss data
loss_df = pd.read_pickle("./grid_spaceganlosses.pkl.gz")

#Plot losses and selection criteria side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

loss_df.plot(ax=ax1,alpha=0.7)
ax1.set_title("Generator and Discriminator loss during training")

gan_metrics["agg_metrics"].plot(ax=ax2)
ax2.set_title("Selection criteria during training")



'''