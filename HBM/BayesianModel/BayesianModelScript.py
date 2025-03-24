# %%
import numpy as np
import pandas as pd
import os
import random
import h5py
import torch
import torch.nn as nn
import jax
import joblib
import arviz as az
from jax import random

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

import matplotlib.pyplot as plt

from pytorch_lightning import LightningModule, Trainer
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from torch.utils.data import DataLoader, TensorDataset

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import corner

garstec_data = r'C:\Users\kiena\Documents\YEAR 4\PROJECT\Data\Garstec_AS09_chiara.hdf5'

sun_numax = 3090

# %% [markdown]
# ## Neural network 

# %%
class GarstecNet(LightningModule):
    def __init__(self, input_dim, output_dim, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),  # First layer
            nn.LeakyReLU(),
            nn.Linear(512, 512),  # Second layer [THIS NEEDS TO BE 512]
            nn.LeakyReLU(),
            nn.Linear(512, 256),  # Third layer [INPUT NEEDS TO BE 512]
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, output_dim)
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)
    
ckpt_path = "best_model_v9-1-Big-SS---epoch=28227-val_loss=0.00008333.ckpt"
input_dim = 7  # Number of input features
output_dim = 5  # Number of output features
model = GarstecNet.load_from_checkpoint(ckpt_path, input_dim=input_dim, output_dim=output_dim)
device = torch.device("cpu")
model.to(device)
model.eval()  

# %% [markdown]
# ## Importing scaling relations

# %%


# %% [markdown]
# ## Extracting data for specific track

# %%
import jax.numpy as jnp

# Plotting for neural network and garstec track: 07298
specific_track_name = 'track01624'

# Retrieve all required inputs for track: 07298
with h5py.File(garstec_data, 'r') as hdf:
    specific_track = hdf['grid']['tracks'][specific_track_name]

    age_07298 = specific_track['age'][:].reshape(-1, 1)
    massini_07298 = specific_track['massini'][:].reshape(-1, 1)
    fehini_07298 = specific_track['FeHini'][:].reshape(-1, 1)
    alphamlt_07298 = specific_track['alphaMLT'][:].reshape(-1, 1)
    yini_07298 = specific_track['yini'][:].reshape(-1, 1)
    eta_07298 = specific_track['eta'][:].reshape(-1, 1)
    alphafe_07298 = specific_track['alphaFe'][:].reshape(-1, 1)

    # Retrieve actual values for plotting
    teff_07298 = specific_track['Teff'][:]
    radius_07298 = specific_track['radPhot'][:]
    dnufit_07298 = specific_track['dnufit'][:]
    FeH_07298 = specific_track['FeH'][:]
    numax_07298 = specific_track['numax'][:]

# Using a single age point from the middle of the ages:

index_07298 = round(len(age_07298)/2)

age_07298_ = age_07298[index_07298]
massini_07298_ = massini_07298[index_07298]
fehini_07298_ = fehini_07298[index_07298]
alphamlt_07298_ = alphamlt_07298[index_07298]
yini_07298_ = yini_07298[index_07298]
eta_07298_ = eta_07298[index_07298]
alphafe_07298_ = alphafe_07298[index_07298]


teff_07298_ = teff_07298[index_07298]
radius_07298_ = radius_07298[index_07298]
dnufit_07298_ = dnufit_07298[index_07298]
FeH_07298_ = FeH_07298[index_07298]
numax_07298_ = numax_07298[index_07298]



# %% [markdown]
# ## Emulate function for Bayesian model

# %%

# Temp, dnu, numax, FeH prediction

class GarstecNet(LightningModule):
    def __init__(self, input_dim, output_dim, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),  # First layer
            nn.LeakyReLU(),
            nn.Linear(512, 512),  # Second layer [THIS NEEDS TO BE 512]
            nn.LeakyReLU(),
            nn.Linear(512, 256),  # Third layer [INPUT NEEDS TO BE 512]
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, output_dim)
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)
    
ckpt_path = "best_model_v9-1-Big-SS---epoch=28227-val_loss=0.00008333.ckpt"
input_dim = 7  # Number of input features
output_dim = 5  # Number of output features
model1 = GarstecNet.load_from_checkpoint(ckpt_path, input_dim=input_dim, output_dim=output_dim)
device = torch.device("cpu")
model1.to(device)
model1.eval()  

scaler_X1 = joblib.load('input_scalerV9.pkl')
scaler_y1 = joblib.load('output_scalerV9.pkl')

state_dict1 = model1.state_dict()
weight1 = [jnp.asarray(param.numpy()) for name, param in state_dict1.items() if "weight" in name]
bias1 = [jnp.asarray(param.numpy()) for name, param in state_dict1.items() if "bias" in name]


def emulate1(x):
    if x.ndim == 1:
        x = x[None, :]  # Convert to shape (1, features) for a single sample

    # Hidden layers
    for i, (w, b) in enumerate(zip(weight1[:-1], bias1[:-1])):
        if x.shape[1] != w.shape[1]:
            raise ValueError(f"Shape mismatch in layer {i}: x.shape[1] ({x.shape[1]}) != w.shape[1] ({w.shape[1]})")
        #print(f"Layer {i}: x shape: {x.shape}, w.T shape: {w.T.shape}, b shape: {b.shape}")
        x = jax.nn.leaky_relu(jnp.dot(x, w.T) + b)

    # Final layer
    if x.shape[1] != weight1[-1].shape[1]:  # Corrected check
        raise ValueError(f"Final layer mismatch: x.shape[1] ({x.shape[1]}) != weight[-1].shape[1] ({weight1[-1].shape[1]})")
    #print(f"Final layer: x shape: {x.shape}, weight[-1].T shape: {weight[-1].T.shape}, bias[-1] shape: {bias[-1].shape}")
    x = jnp.dot(x, weight1[-1].T) + bias1[-1]
    return x

# %%
# Radius prediction

# Lightning Module
class GarstecNet(LightningModule):
    def __init__(self, input_dim, output_dim, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        
        # Define MSE loss
        self.loss_fn = nn.MSELoss()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),  # First layer
            nn.LeakyReLU(),
            nn.Linear(256, 256),  # Second layer
            nn.LeakyReLU(),
            nn.Linear(256, 128),  # Third layer
            nn.LeakyReLU(),
            nn.Linear(128, 128),  # Fourth layer
            nn.LeakyReLU(),
            nn.Linear(128, output_dim)  # Output layer
        )

    def forward(self, x):
        return self.model(x)

    
ckpt_path = "best_model_v14-RadSingle---epoch=19966-val_loss=0.00141896.ckpt"
input_dim = 7  # Number of input features
output_dim = 1  # Number of output features
model2 = GarstecNet.load_from_checkpoint(ckpt_path, input_dim=input_dim, output_dim=output_dim)
device = torch.device("cpu")
model2.to(device)
model2.eval()  

scaler_X2 = joblib.load('input_scalerV14.pkl')
scaler_y2 = joblib.load('output_scalerV14.pkl')

state_dict2 = model2.state_dict()
weight2 = [jnp.asarray(param.numpy()) for name, param in state_dict2.items() if "weight" in name]
bias2 = [jnp.asarray(param.numpy()) for name, param in state_dict2.items() if "bias" in name]


def emulate2(x):
    if x.ndim == 1:
        x = x[None, :]  # Convert to shape (1, features) for a single sample

    # Hidden layers
    for i, (w, b) in enumerate(zip(weight2[:-1], bias2[:-1])):
        if x.shape[1] != w.shape[1]:
            raise ValueError(f"Shape mismatch in layer {i}: x.shape[1] ({x.shape[1]}) != w.shape[1] ({w.shape[1]})")
        #print(f"Layer {i}: x shape: {x.shape}, w.T shape: {w.T.shape}, b shape: {b.shape}")
        x = jax.nn.leaky_relu(jnp.dot(x, w.T) + b)

    # Final layer
    if x.shape[1] != weight2[-1].shape[1]:  # Corrected check
        raise ValueError(f"Final layer mismatch: x.shape[1] ({x.shape[1]}) != weight[-1].shape[1] ({weight2[-1].shape[1]})")
    #print(f"Final layer: x shape: {x.shape}, weight[-1].T shape: {weight[-1].T.shape}, bias[-1] shape: {bias[-1].shape}")
    x = jnp.dot(x, weight2[-1].T) + bias1[-1]
    return x

# %%
obs_err = {
    'teff_err': [70],
    'rad_err': [3],
    'dnu_err': [0.1],
    'feh_err': [0.1],
    'numax_err': [0.5]
}

teff_obs = teff_07298_ + np.random.randn() * obs_err['teff_err'][0]
rad_obs = radius_07298_ + np.random.randn() * obs_err['rad_err'][0]
dnu_obs = dnufit_07298_ + np.random.randn() * obs_err['dnu_err'][0]
FeH_obs = FeH_07298_ + np.random.randn() * obs_err['feh_err'][0]
numax_obs = numax_07298_ * sun_numax + np.random.randn() * obs_err['numax_err'][0]

obs = {
    'teff': [teff_obs],
    'rad': [rad_obs],
    'dnu': [dnu_obs],
    'feh': [FeH_obs],
    'numax': [numax_obs],
}



# %% [markdown]
# ## Bayesian model

# %%
def Bmodel(obs=None):
    # Define priors
    massini_ = numpyro.deterministic("massini_", 0.8 * numpyro.sample("massini_s", dist.Beta(3, 6)) + 0.7)
    tau_hat = numpyro.deterministic("tau_hat", 4 * numpyro.sample("tau_hat_s", dist.Beta(1.25, 2)) + 1) 

    # Calculate values for joint prior: 
    tau_ms = 3500 * (massini_**-3.15)
    ages_ = numpyro.deterministic("ages_", jnp.minimum(tau_hat * tau_ms, 20000))  # Cap at 20000 to stay in grid bounds

    # Rest of priors:
    alphamlt_ = numpyro.deterministic("alphamlt_", 0.8 * numpyro.sample("alphamlt_s", dist.Beta(2, 2)) + 1.5)
    yini_ = numpyro.deterministic("yini_", 0.13 * numpyro.sample("yini_s", dist.Beta(2, 2)) + 0.22)
    eta_ = numpyro.deterministic("eta_", 0.3 * numpyro.sample("eta_s", dist.Beta(2, 2)))
    alphafe_ = numpyro.deterministic("alphafe_", 0.8 * numpyro.sample("alphafe_s", dist.Beta(2, 2)) - 0.2) 
    fehini_ = numpyro.deterministic("fehini_", 2.2 * numpyro.sample("fehini_s", dist.Beta(2, 2)) - 2)

    # Prepare input features for model prediction
    epsilon = 1e-10
    log_vars_inputs = [ages_, massini_, alphamlt_, eta_, yini_] 
    log_transformed_inputs = [jnp.log10(jnp.maximum(var, epsilon)) for var in log_vars_inputs]
    x = jnp.hstack(log_transformed_inputs + [fehini_, alphafe_])

    # Scale inputs using saved scaler
    x1_scaled = (x - scaler_X1.mean_) / scaler_X1.scale_
    
    x2_scaled = (x - scaler_X2.mean_) / scaler_X2.scale_

    # Emulate using neural network
    y1_scaled = emulate1(x1_scaled)
    y2_scaled = emulate2(x2_scaled)

    # Descale outputs
    y1 = y1_scaled * jnp.array(scaler_y1.scale_) + jnp.array(scaler_y1.mean_)
    y2 = y2_scaled * jnp.array(scaler_y2.scale_) + jnp.array(scaler_y2.mean_)

    # Extract predictions (applying inverse transformations)
    teff = numpyro.deterministic("teff", jnp.power(10.0, y1[..., 0])) 
    rad = numpyro.deterministic("rad", y2) 
    dnu = numpyro.deterministic("dnu", jnp.power(10.0, y1[..., 2]))
    numax = numpyro.deterministic("numax", jnp.power(10.0, y1[..., 3])) * sun_numax
    feh = numpyro.deterministic("feh", y1[..., 4])  # No transformation needed
    
    # Observational likelihoods (if observations provided)
    if obs is not None:
        numpyro.sample("teff_obs", dist.StudentT(5, teff, obs_err['teff_err'][0]), obs=obs['teff'][0])
        numpyro.sample("rad_obs", dist.StudentT(5, rad, obs_err['rad_err'][0]), obs=obs['rad'][0])
        numpyro.sample("dnu_obs", dist.StudentT(5, dnu, obs_err['dnu_err'][0]), obs=obs['dnu'][0])
        numpyro.sample("feh_obs", dist.StudentT(5, feh, obs_err['feh_err'][0]), obs=obs['feh'][0])
        numpyro.sample("numax_obs", dist.StudentT(5, numax, obs_err['numax_err'][0]), obs=obs['numax'][0])

# %% [markdown]
# ## Prior predictive

# %%
from numpyro.infer import Predictive

from jax import random

prior_predictive = Predictive(Bmodel, num_samples=1000)
samples = prior_predictive(jax.random.PRNGKey(0))

# %%
# fig = corner.corner(samples, var_names=['teff', 'lum', 'dnu', 'feh', 'numax'], show_titles=True)
# plt.show()

# %%
from jax import random

nuts_kernel = NUTS(Bmodel)

prior_mcmc = MCMC(nuts_kernel, num_samples=8000, num_warmup=8000)
rng_key = random.PRNGKey(0)
prior_mcmc.run(rng_key, obs=None)

posterior_samples = prior_mcmc.get_samples()

# %%
import arviz as az

prior_trace = az.from_numpyro(prior_mcmc)
az.plot_trace(prior_trace, var_names=['ages_', 'massini_', 'alphamlt_', 'eta_', 'yini_', 'fehini_',
                                       'alphafe_', 'teff', 'lum', 'feh', 'dnu', 'numax'], compact=False, figsize=(15, 18));  

plt.subplots_adjust(hspace=1)  

# %%
truths_prior = truth_values = [age_07298[0].item(), massini_07298[0].item(),
                                alphamlt_07298[0].item(), eta_07298[0].item(),
                                  yini_07298[0].item(),  fehini_07298[0].item(),
                                    alphafe_07298[0].item()]
#print(truth_values)

corner.corner(prior_trace, var_names=['ages_', 'massini_', 'alphamlt_', 'eta_', 'yini_', 'fehini_', 'alphafe_'], 
             truths = truths_prior);


# %% [markdown]
# ## Running model and Posterior plotting 

# %%
from numpyro.infer.initialization import init_to_median
from jax import random

nuts = NUTS(Bmodel, target_accept_prob=0.8, init_strategy=init_to_median, find_heuristic_step_size=True)
mcmc = MCMC(nuts, num_warmup=2000, num_samples=2000, num_chains=2) # between 1000 and 4000 for testing 
rng = random.PRNGKey(0)
rng, key = random.split(rng)

mcmc.run(key, obs=obs)

# %%
import arviz as az

trace = az.from_numpyro(mcmc)

az.summary(trace)

# %%
az.plot_trace(trace);

# %%
# Setting truth values for corner plot
truth_values = [age_07298_.item(), massini_07298[0].item(), 
                alphamlt_07298[0].item(), eta_07298[0].item(),
                yini_07298[0].item(),  fehini_07298[0].item(),
                alphafe_07298[0].item()]
print(truth_values)
import corner

corner.corner(trace, var_names=['ages_', 'massini_', 'alphamlt_', 'eta_', 'yini_', 'fehini_', 'alphafe_'], 
             truths = truth_values);

# %%
figure = corner.corner(prior_trace, var_names=['ages_', 'massini_', 'alphamlt_', 'eta_', 'yini_', 'fehini_', 'alphafe_'],
              color='r');

corner.corner(trace, var_names=['ages_', 'massini_', 'alphamlt_', 'eta_', 'yini_', 'fehini_', 'alphafe_'], 
            fig=figure,
            truths = truth_values);
plt.show()

# %%
from sklearn.preprocessing import MinMaxScaler


# Extract the variables from the trace
ages_samples = trace.posterior['ages_'].values.flatten()
mass_samples = trace.posterior['massini_'].values.flatten()
mlt_samples = trace.posterior['alphamlt_'].values.flatten()
eta_samples = trace.posterior['eta_'].values.flatten()
yini_samples = trace.posterior['yini_'].values.flatten()
fehini_samples = trace.posterior['fehini_'].values.flatten()
fe_samples = trace.posterior['alphafe_'].values.flatten()

closest_index = np.argmin(np.abs(mass_samples - massini_07298_))

length = len(age_07298)
      
ages_indices = []

for i in range(length):
    # Find the index in ages_samples closest to the current age value
    closest_index = np.argmin(np.abs(ages_samples - age_07298[i]))
    ages_indices.append(closest_index)


# Select the next `length` values after the min_age index
ages_subsample = np.sort(ages_samples[ages_indices]).reshape(-1, 1)
mass_subsample = np.full(length, mass_samples[closest_index]).reshape(-1, 1) 
mlt_subsample = np.full(length, np.mean(mlt_samples)).reshape(-1, 1)
eta_subsample = np.full(length, np.mean(eta_samples)).reshape(-1, 1) 
yini_subsample = np.full(length, np.mean(yini_samples)).reshape(-1, 1) 
fehini_subsample = np.full(length, np.mean(fehini_samples)).reshape(-1, 1) 
fe_subsample = np.full(length, np.mean(fe_samples)).reshape(-1, 1) 

epsilon = 1e-10
log10_hbm_inputs = [age_07298, mass_subsample, mlt_subsample, eta_subsample, yini_subsample] 
log10_transformed_hbm_inputs = [np.log10(np.maximum(data, epsilon)) for data in log10_hbm_inputs]

# Combine log-transformed inputs with raw `fehini` and `alpha fe`
features = np.hstack(log10_transformed_hbm_inputs + [fehini_subsample, fe_subsample])

features_numpy = np.array(features)  # Convert to numpy for scaling

features_scaled = scaler_X.transform(features_numpy)  # Apply scaler

# Convert scaled inputs to PyTorch tensor

features_tensor = torch.FloatTensor(features_scaled)


# %%
print(mass_subsample[0],
mlt_subsample[0],
eta_subsample[0],
yini_subsample[0],
fehini_subsample[0],
fe_subsample[0])

print(massini_07298_, alphamlt_07298_, eta_07298_, yini_07298_, fehini_07298_, alphafe_07298_)

# %%
model.eval()
with torch.no_grad():
    predictions_hbm = model(features_tensor).numpy()  # Make predictions
    predictions_hbm1 = scaler_y.inverse_transform(predictions_hbm)  # Inverse transform

# Extract predicted `Teff` and `Luminosity`
hbm_teff = 10**predictions_hbm1[:, 0]  # Inverse log10 transformation
hbm_luminosity = 10**predictions_hbm1[:, 1]

# Log-transform true values for plotting
log_actual_teff = np.log10(teff_07298)
log_actual_luminosity = np.log10(luminosity_07298)

# Log-transform predicted values for plotting
log_hbm_teff = np.log10(hbm_teff)
log_hbm_luminosity = np.log10(hbm_luminosity)

# Plot HR Diagram for the selected track
plt.figure(figsize=(5.4, 4))  # Slightly less than half-width of A4 landscape
plt.plot(log_actual_teff, log_actual_luminosity, label='True Values', color='blue', marker='o', markersize=1, linestyle='-', alpha=0.7)
plt.plot(log_hbm_teff, log_hbm_luminosity, label='Predicted Values', color='red', marker='x', markersize=1, linestyle='--', alpha=0.7)
plt.gca().invert_xaxis()  # Effective temperature is plotted in reverse
plt.xlabel("Log Effective Temperature ($\log(T_\mathrm{eff})$)")
plt.ylabel(f"Log Luminosity ($\log(L_\odot)$)")
plt.title(f"HR Diagram for: {specific_track_name}")
plt.legend()
plt.show()


