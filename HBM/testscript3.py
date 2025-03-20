import jax
import jax.numpy as jnp
import json
import os
from itertools import product
from pathlib import Path
from typing import List, Tuple, Union
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
import arviz as az
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from jax import random
from numpyro.infer.initialization import init_to_median

numpyro.util.set_host_device_count(8)

df = pd.read_csv(r'HBM/tailo2022table.csv') 
df_first_30 = df[:32]

Teff_M4 = jnp.array(df_first_30["Teff"])
L_M4 = jnp.array(df_first_30["L"])
errL_M4 = jnp.array(df_first_30["errL"])
numax_M4 = jnp.array(df_first_30["numax"])
errnumax_M4 = jnp.array(df_first_30["errnumax"])
Deltanu_M4 = jnp.array(df_first_30["Deltanu"])
errDeltanu_M4 = jnp.array(df_first_30["errDeltanu"])

class PCANN(tf.keras.layers.Layer):
    """
    Inverse PCA layer for tensorflow neural network

    Usage:
        - Define dictionary of custom objects containing Inverse PCA
        - Use arguments of PCA mean and components from PCA of output parameters for inverse PCA (found in JSON dict)

    Example:

    > f = open("pcann_info.json")
    >
    > data = json.load(f)
    >
    > pca_comps = np.array(data["pca_comps"])
    > pca_mean = np.array(data["pca_mean"])
    >
    > custom_objects = {"InversePCA": InversePCA(pca_comps, pca_mean)}
    > pcann_model = tf.keras.models.load_model("pcann_name.h5", custom_objects=custom_objects)

    """

    def __init__(self, pca_comps, pca_mean, **kwargs):
        super(PCANN, self).__init__()
        self.pca_comps = pca_comps
        self.pca_mean = pca_mean

    def call(self, x):
        y = tf.tensordot(x, np.float32(self.pca_comps), 1) + np.float32(self.pca_mean)
        return y

    def get_config(self):
        config = super().get_config().copy()
        config.update({"pca_comps": self.pca_comps, "pca_mean": self.pca_mean})
        return config
    

class WMSE(tf.keras.losses.Loss):
    """
    Weighted Mean Squared Error Loss Function for tensorflow neural network

    Usage:
        - Define list of weights with len = labels
        - Use weights as arguments - no need to square, this is handled in-function
        - Typical usage - defining target precision on outputs for the network to achieve, weights parameters in loss calculation to force network to focus on parameters with unc >> weight.

    """

    def __init__(self, weights, name="WMSE", **kwargs):
        super(WMSE, self).__init__()
        self.weights = np.float32(weights)

    def call(self, y_true, y_pred):
        loss = ((y_true - y_pred) / (self.weights)) ** 2
        return tf.math.reduce_mean(loss)

    def get_config(self):
        config = super().get_config().copy()
        config.update({"weights": self.weights})
        return config


def WMSE_metric(y_true, y_pred):
    metric = ((y_true - y_pred) / (weights)) ** 2
    return tf.reduce_mean(metric)

run_name = "HBM/smiley_stingray"


emulatorpath = f"./{run_name}/"

logfile = f"log_{run_name}.json"
pcafile = f"pca_{run_name}.json"
checkpointfile = f"{run_name}_checkpoint.h5"
historyfile = f"history_{run_name}.json"

with open(os.path.join(emulatorpath, logfile), "r") as fp:
    data = json.load(fp)

    gridpath = data["gridpath"]
    gridfile = data["gridfile"]
    grid = os.path.join(gridpath, gridfile)

    seed = data["seed"]
    n_components = data["n_components"]

    batch_size_exp = data["batch_size_exp"]
    epochs = data["epochs"]
    test_size = data["test_size"]
    fractrain = data["fractrain"]

    inputs = data["inputs"]
    classical_outputs = data["classical_outputs"]
    nmin = data["nmin"]
    nmax = data["nmax"]

astero_outputs = [f"nu_0_{i+1}" for i in range(nmin - 1, nmax)]
outputs = classical_outputs + astero_outputs

def load_pca_components(emulatorpath: str, pcafile: str) -> (np.array, np.array):
    with open(os.path.join(emulatorpath, pcafile), "r") as fp:
        data = json.load(fp)
        pca_comps = np.array(data["pca_comps"])
        pca_mean = np.array(data["pca_mean"])
    return pca_comps, pca_mean


def get_weights_and_biases(tf_model: Model) -> (list, list):
    weights = list(map(jnp.asarray, tf_model.weights[::2]))
    biases = list(map(jnp.asarray, tf_model.weights[1::2]))
    return weights, biases


def load_tf_model(
    emulatorpath: str,
    checkpointfile: str,
    pcafile: str | None = None,
    pca_comps: np.ndarray | None = None,
    pca_mean: np.ndarray | None = None,
    n: int = 25,
):
    if pca_comps is None or pca_mean is None:
        assert pcafile is not None
        pca_comps, pca_mean = load_pca_components(emulatorpath, pcafile)

    custom_objects = {
        "PCANN": PCANN(pca_comps, pca_mean),
        "WMSE": WMSE(np.ones(n)),
    }

    tf_model = tf.keras.models.load_model(
        os.path.join(emulatorpath, checkpointfile), custom_objects=custom_objects
    )
    return tf_model


def load_emulator(run_name: str, emulatorpath: str, checkpointfile: str, pcafile: str):
    pca_comps, pca_mean = load_pca_components(emulatorpath, pcafile)

    tf_model = load_tf_model(
        emulatorpath,
        checkpointfile,
        pca_comps=pca_comps,
        pca_mean=pca_mean,
    )
    weights, biases = get_weights_and_biases(tf_model)

    stem_map = [0, 1]
    ctine_map = [-5, -3, -1]
    atine_map = [-10, -9, -8, -7, -6, -4, -2]

    emulator = (
        weights,
        biases,
        stem_map,
        ctine_map,
        atine_map,
        pca_comps,
        pca_mean,
    )
    return emulator

logcols = [
    "initial_mass",
    "alphaMLT",
    "radius",
    "luminosity",
    "mass",
]
logcols += ["error_" + col for col in logcols]


def scale(
    data: Union[pd.DataFrame, np.ndarray],
    logcols: List[str] = logcols,
    col_names: List[str] | None = None,
    verbose: bool = False,
) -> Union[pd.DataFrame, np.ndarray]:
    if isinstance(data, np.ndarray):
        if col_names is None:
            raise ValueError("col_names must be provided when data is a NumPy array.")
        df_unnorm = pd.DataFrame(data, columns=col_names)
    else:
        df_unnorm = data

    if col_names is None:
        col_names = df_unnorm.columns
        cols = df_unnorm.values.T
    else:
        cols = data.T

    df_norm = df_unnorm.copy()
    for col_name, col in zip(col_names, cols):
        if col_name in logcols:
            if verbose:
                print(f"{col_name} scaled with log10")
            df_norm[col_name] = np.log10(col)
        elif col_name in ["initial_y", "error_initial_y"]:
            if verbose:
                print(f"{col_name} scaled by multiply with 4 and log10")
            df_norm[col_name] = np.log10(col * 4)
        elif col_name in ["age", "error_age"]:
            if verbose:
                print(f"{col_name} scaled by dividing with 1000 and then log10")
            df_norm[col_name] = np.log10(col / 1000)
        else:
            if verbose:
                print(f"{col_name} not scaled")

    if isinstance(data, np.ndarray):
        return df_norm.values
    return df_norm


def descale(
    data: Union[pd.DataFrame, np.ndarray],
    logcols: List[str] = logcols,
    col_names: List[str] | None = None,
    verbose: bool = False,
) -> Union[pd.DataFrame, np.ndarray]:
    if isinstance(data, np.ndarray):
        if col_names is None:
            raise ValueError("col_names must be provided when data is a NumPy array.")
        df_norm = pd.DataFrame(data, columns=col_names)
    else:
        df_norm = data

    if col_names is None:
        col_names = df_norm.columns
        cols = df_norm.values.T

    df_unnorm = df_norm.copy()
    for col_name, col in zip(col_names, cols):
        if col_name in logcols:
            if verbose:
                print(f"{col_name} descaled using inverse log10")
            df_unnorm[col_name] = 10 ** (col)
        elif col_name == "initial_y":
            if verbose:
                print(f"{col_name} descaled by inverse log10 and then divide by 4")
            df_unnorm[col_name] = (10 ** (col)) / 4
        elif col_name == "age":
            if verbose:
                print(
                    f"{col_name} descaled by inverse log10 and then multiply with 1000"
                )
            df_unnorm[col_name] = (10 ** (col)) * 1000
        else:
            if verbose:
                print(f"{col_name} not descaled")

    if isinstance(data, np.ndarray):
        return df_unnorm.values
    return df_unnorm

def call_emulator(
    input_norm: np.ndarray,
    emulator: Tuple[
        np.ndarray, np.ndarray, list[int], list[int], list[int], np.ndarray, np.ndarray
    ],
    scale_dimensions: List[str] | None = None,
) -> jax.Array:
    stem = input_norm

    (weights, biases, stem_map, ctine_map, atine_map, pca_comps, pca_mean) = emulator

    for index in stem_map:
        stem = jax.nn.elu(jnp.dot(stem, weights[index]) + biases[index])
    xx = jnp.copy(stem)

    for i, cindex in enumerate(ctine_map[:-1]):
        if i == 0:
            ctine = jax.nn.elu(jnp.dot(stem, weights[cindex]) + biases[cindex])
        else:
            ctine = jax.nn.elu(jnp.dot(ctine, weights[cindex]) + biases[cindex])
    ctine_out = jnp.dot(ctine, weights[ctine_map[-1]]) + biases[ctine_map[-1]]

    for i, aindex in enumerate(atine_map[:-1]):
        if i == 0:
            atine = jax.nn.elu(jnp.dot(stem, weights[aindex]) + biases[aindex])
        else:
            atine = jax.nn.elu(jnp.dot(atine, weights[aindex]) + biases[aindex])
    atine_out = jnp.dot(atine, weights[atine_map[-1]]) + biases[atine_map[-1]]
    atine_out = jnp.dot(atine_out, pca_comps) + pca_mean

    out_norm = jnp.concatenate((ctine_out, atine_out), axis=-1)
    return out_norm

def compute_meh(feh, alphaFe):
    falpha = 10 ** alphaFe
    return feh + jnp.log10(0.694 * falpha + 0.306)

emulator = load_emulator(run_name, emulatorpath, checkpointfile, pcafile)

obs = {
    'teff': Teff_M4,
    'lum': L_M4,
    'dnu': Deltanu_M4,
    'numax': numax_M4
}

obs_err = {
    'teff_err': jnp.full_like(Teff_M4, 70),
    'lum_err': errL_M4,
    'dnu_err': errDeltanu_M4,
    'numax_err': errnumax_M4
}

n_stars = 32

import jax 
num_chains = jax.local_device_count()

# Testing emulator:

input_norm = jnp.stack([-0.04963513, -0.65428674, 0.2, 0.01944897, 0.19087858, 0.0628418, 0.8292411], axis=-1)
output_norm = call_emulator(input_norm=input_norm, emulator=emulator)

radius_test = 10 ** output_norm[0]
luminosity_test = 10 ** output_norm[1]
teff_test = 5772 * ((radius_test ** (-2) * luminosity_test) ** (0.25))

# Compute numax for the sole purpose of being a scale in the surface correction
dnu_test = jnp.median(jnp.diff(output_norm[len(classical_outputs) :]))
numax_test = (dnu_test / 0.263) ** (1 / 0.772)  # Stello et al

print('emu', luminosity_test)
print('emu', teff_test)


def Bmodel(obs=None):
    yini_ = numpyro.deterministic("yini_", 0.13 * numpyro.sample("yini_s", dist.Beta(2, 7)) + 0.248)
    yini_scaled = jnp.log10(yini_*4) # For Amalie's NN
    alphamlt_ = numpyro.deterministic("alphamlt_", 0.8 * numpyro.sample("alphamlt_s", dist.Beta(5, 5)) + 1.5)
    eta_ = numpyro.deterministic("eta_", 0.3 * numpyro.sample("eta_s", dist.Beta(2, 7))) 
    fehini_ = numpyro.deterministic("fehini_", 0.1 * numpyro.sample("fehini_s", dist.Beta(2, 2)) - 1.12)
    afe_meh = numpyro.deterministic("afe_meh", 0.25 * numpyro.sample("afe_s", dist.Beta(5, 5)) + 0.2)
    meh_ =  numpyro.deterministic("meh_", compute_meh(fehini_, afe_meh))
    M4_age = numpyro.deterministic("M4_age", 4.0 * numpyro.sample("M4_age_s", dist.Beta(5, 5)) + 9)
    M4_age_spread = 0.1 
    
    with numpyro.plate("star", n_stars):
        # Define priors
        massini_ = numpyro.deterministic("massini_", 0.8 * numpyro.sample("massini_s", dist.Beta(3, 6)) + 0.7)
        alphafe_ = numpyro.deterministic("alphafe_", 0.25 * numpyro.sample("alphafe_s", dist.Beta(5, 5)) + 0.2)
        age = numpyro.deterministic("age", M4_age_spread * numpyro.sample("age_s", dist.Normal(0, 1)) + M4_age)

    yini_true = jnp.repeat(yini_scaled, n_stars)
    alphamlt_true = jnp.repeat(alphamlt_, n_stars)
    eta_true = jnp.repeat(eta_, n_stars)
    meh_true = jnp.repeat(meh_, n_stars)
    norm_age = age/M4_age

        
    x = jnp.stack([jnp.log10(massini_), meh_true, alphafe_, yini_true, 
                    jnp.log10(alphamlt_true), eta_true, norm_age], axis=-1)
    
    y = call_emulator(input_norm=x, emulator=emulator)

    rad = numpyro.deterministic("rad", 10 ** y[..., 0])
    lum = numpyro.deterministic("luminosity", 10 ** y[..., 1])
    teff = numpyro.deterministic("teff", 5772 * ((rad ** (-2) * lum) ** (0.25)))
    
    # Compute numax for the sole purpose of being a scale in the surface correction
    dnu = jnp.median(jnp.diff(y[..., len(classical_outputs) :], axis=-1), axis=-1) # dnu = jnp.median(jnp.diff(y[len(classical_outputs) :]))
    numax = (dnu / 0.263) ** (1 / 0.772)  # Stello et al
    alphafe = alphafe_
        
    # Observational likelihoods
    if obs is not None:
        num_obs = len(obs['teff'])
            
        # Sample observations
        numpyro.sample("teff_obs", dist.StudentT(5, teff, obs_err['teff_err']), obs=obs['teff'])
        numpyro.sample("lum_obs", dist.StudentT(5, lum, obs_err['lum_err']), obs=obs['lum'])
        numpyro.sample("dnu_obs", dist.StudentT(5, dnu, obs_err['dnu_err']), obs=obs['dnu'])
        numpyro.sample("numax_obs", dist.StudentT(5, numax, obs_err['numax_err']), obs=obs['numax'])
        numpyro.sample("alphafe_obs", dist.StudentT(5, alphafe, obs_err['alphafe_err']), obs=obs['alphafe'])

nuts = NUTS(Bmodel, target_accept_prob=0.8, init_strategy=init_to_median, find_heuristic_step_size=True)
mcmc = MCMC(nuts, num_warmup=4000, num_samples=2000, num_chains=8) # between 1000 and 4000 for testing 
rng = random.PRNGKey(0)
rng, key = random.split(rng)

mcmc.run(key, obs=obs)

trace = az.from_numpyro(mcmc)

trace.to_netcdf("HBMtest1.nc")