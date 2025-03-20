import jax
import jax.numpy as jnp
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional, Any
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from tensorflow.keras.models import Model
import arviz as az
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from jax import random
from numpyro.infer.initialization import init_to_median

# Configure NumPyro to use all available devices
numpyro.util.set_host_device_count(4)

# Configuration constants
N_COMPONENTS = 25
N_STARS = 32
NUM_CHAINS = 4
RANDOM_SEED = 0

# Log columns that need scaling
LOG_SCALE_COLUMNS = [
    "initial_mass",
    "alphaMLT",
    "radius",
    "luminosity",
    "mass",
]
LOG_SCALE_COLUMNS += [f"error_{col}" for col in LOG_SCALE_COLUMNS]


class PCANN(tf.keras.layers.Layer):
    """
    Inverse PCA layer for tensorflow neural network.
    
    This layer performs inverse PCA transformation on network outputs.
    
    Args:
        pca_comps: PCA components matrix
        pca_mean: PCA mean vector
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
    Weighted Mean Squared Error Loss Function for tensorflow neural network.
    
    Applies weights to the squared error terms to emphasize certain parameters.
    
    Args:
        weights: Array of weights for each output dimension
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
    """Metric function for weighted mean squared error."""
    # Note: 'weights' should be defined before using this function
    metric = ((y_true - y_pred) / (weights)) ** 2
    return tf.reduce_mean(metric)


def load_pca_components(pcafile: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load PCA components and mean from a JSON file.
    
    Args:
        pcafile: Path to the PCA components JSON file
        
    Returns:
        Tuple of (pca_components, pca_mean)
    """
    try:
        with open(pcafile, "r") as fp:
            data = json.load(fp)
            pca_comps = np.array(data["pca_comps"])
            pca_mean = np.array(data["pca_mean"])
        return pca_comps, pca_mean
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to load PCA components from {pcafile}: {str(e)}")


def get_weights_and_biases(tf_model: Model) -> Tuple[list, list]:
    """
    Extract weights and biases from a TensorFlow model.
    
    Args:
        tf_model: TensorFlow model
        
    Returns:
        Tuple of (weights_list, biases_list)
    """
    weights = list(map(jnp.asarray, tf_model.weights[::2]))
    biases = list(map(jnp.asarray, tf_model.weights[1::2]))
    return weights, biases


def load_tf_model(
    checkpointfile: str, 
    pca_comps: Optional[np.ndarray] = None, 
    pca_mean: Optional[np.ndarray] = None, 
    pcafile: Optional[str] = None,
    n_outputs: int = 25
) -> Model:
    """
    Load a TensorFlow model with custom objects.
    
    Args:
        checkpointfile: Path to the model checkpoint file
        pca_comps: PCA components matrix (optional if pcafile is provided)
        pca_mean: PCA mean vector (optional if pcafile is provided)
        pcafile: Path to PCA components file (optional if pca_comps and pca_mean are provided)
        n_outputs: Number of outputs for the WMSE loss
        
    Returns:
        Loaded TensorFlow model
    """
    if (pca_comps is None or pca_mean is None) and pcafile is not None:
        pca_comps, pca_mean = load_pca_components(pcafile)
    elif pca_comps is None or pca_mean is None:
        raise ValueError("Either pca_comps and pca_mean or pcafile must be provided")

    custom_objects = {
        "PCANN": PCANN(pca_comps, pca_mean),
        "WMSE": WMSE(np.ones(n_outputs)),
    }

    try:
        tf_model = tf.keras.models.load_model(checkpointfile, custom_objects=custom_objects)
        return tf_model
    except (IOError, tf.errors.OpError) as e:
        raise RuntimeError(f"Failed to load model from {checkpointfile}: {str(e)}")


def load_emulator(
    checkpointfile: str, 
    pcafile: str
) -> Tuple[list, list, list, list, list, np.ndarray, np.ndarray]:
    """
    Load the emulator model and prepare components for inference.
    
    Args:
        checkpointfile: Path to the model checkpoint file
        pcafile: Path to the PCA components file
        
    Returns:
        Tuple containing all components needed for the emulator
    """
    # Load PCA components
    pca_comps, pca_mean = load_pca_components(pcafile)

    # Load TensorFlow model
    tf_model = load_tf_model(checkpointfile, pca_comps, pca_mean)
    
    # Extract model parameters
    weights, biases = get_weights_and_biases(tf_model)

    # Define layer mappings for different paths in the network
    stem_map = [0, 1]
    ctine_map = [-5, -3, -1]
    atine_map = [-10, -9, -8, -7, -6, -4, -2]

    # Package emulator components
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


def scale(
    data: Union[pd.DataFrame, np.ndarray],
    logcols: List[str] = LOG_SCALE_COLUMNS,
    col_names: Optional[List[str]] = None,
    verbose: bool = False,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Scale data for neural network input.
    
    Args:
        data: Input data as DataFrame or numpy array
        logcols: List of columns to apply log10 scaling
        col_names: Column names when data is numpy array
        verbose: Whether to print scaling operations
        
    Returns:
        Scaled data in the same format as input
    """
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
    logcols: List[str] = LOG_SCALE_COLUMNS,
    col_names: Optional[List[str]] = None,
    verbose: bool = False,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Reverse scaling of data from neural network output.
    
    Args:
        data: Input data as DataFrame or numpy array
        logcols: List of columns to apply inverse log10 scaling
        col_names: Column names when data is numpy array
        verbose: Whether to print descaling operations
        
    Returns:
        Descaled data in the same format as input
    """
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
    classical_outputs_len: int,
) -> jax.Array:
    """
    Call the emulator with normalized inputs.
    
    Args:
        input_norm: Normalized input array
        emulator: Tuple of emulator components
        classical_outputs_len: Number of classical outputs
        
    Returns:
        Emulator output array
    """
    stem = input_norm

    (weights, biases, stem_map, ctine_map, atine_map, pca_comps, pca_mean) = emulator

    # Stem network
    for index in stem_map:
        stem = jax.nn.elu(jnp.dot(stem, weights[index]) + biases[index])
    
    # Classical parameters branch
    ctine = stem
    for i, cindex in enumerate(ctine_map[:-1]):
        if i == 0:
            ctine = jax.nn.elu(jnp.dot(stem, weights[cindex]) + biases[cindex])
        else:
            ctine = jax.nn.elu(jnp.dot(ctine, weights[cindex]) + biases[cindex])
    ctine_out = jnp.dot(ctine, weights[ctine_map[-1]]) + biases[ctine_map[-1]]

    # Asteroseismic parameters branch
    atine = stem
    for i, aindex in enumerate(atine_map[:-1]):
        if i == 0:
            atine = jax.nn.elu(jnp.dot(stem, weights[aindex]) + biases[aindex])
        else:
            atine = jax.nn.elu(jnp.dot(atine, weights[aindex]) + biases[aindex])
    atine_out = jnp.dot(atine, weights[atine_map[-1]]) + biases[atine_map[-1]]
    atine_out = jnp.dot(atine_out, pca_comps) + pca_mean

    # Concatenate outputs
    out_norm = jnp.concatenate((ctine_out, atine_out), axis=-1)
    return out_norm


def compute_meh(feh: float, alphaFe: float) -> float:
    """
    Compute metallicity [M/H] from [Fe/H] and [alpha/Fe].
    
    Args:
        feh: Iron abundance [Fe/H]
        alphaFe: Alpha element abundance [alpha/Fe]
        
    Returns:
        Metallicity [M/H]
    """
    falpha = 10 ** alphaFe
    return feh + jnp.log10(0.694 * falpha + 0.306)


def setup_bayesian_model(
    emulator: Tuple,
    n_stars: int,
    classical_outputs: List[str],
):
    """
    Set up the Bayesian model for inference.
    
    Args:
        emulator: Emulator components
        n_stars: Number of stars in the dataset
        classical_outputs: List of classical output parameter names
        
    Returns:
        Bayesian model function
    """
    def bayesian_model(obs=None):
        # Global parameters (common to all stars)
        yini_ = numpyro.deterministic("yini_", 0.13 * numpyro.sample("yini_s", dist.Beta(2, 7)) + 0.248)
        yini_scaled = jnp.log10(yini_ * 4)  # For network input
        
        alphamlt_ = numpyro.deterministic("alphamlt_", 0.8 * numpyro.sample("alphamlt_s", dist.Beta(5, 5)) + 1.5)
        eta_ = numpyro.deterministic("eta_", 0.3 * numpyro.sample("eta_s", dist.Beta(2, 7)))
        alphafe_ = numpyro.deterministic("alphafe_", 0.4 * numpyro.sample("alphafe_s", dist.Beta(5, 5)) - 0.0)
        fehini_ = numpyro.deterministic("fehini_", 2.2 * numpyro.sample("fehini_s", dist.Beta(2, 2)) - 2)
        meh_ = numpyro.deterministic("meh_", compute_meh(fehini_, alphafe_))
        ages_norm_ = numpyro.deterministic("ages_norm_", numpyro.sample("ages_norm_s", dist.Beta(2, 2)))
        
        # Per-star parameters
        with numpyro.plate("star", n_stars):
            massini_ = numpyro.deterministic("massini_", 0.8 * numpyro.sample("massini_s", dist.Beta(3, 6)) + 0.7)
        
        # Repeat global parameters for all stars
        yini_true = jnp.repeat(yini_scaled, n_stars)
        alphamlt_true = jnp.repeat(alphamlt_, n_stars)
        eta_true = jnp.repeat(eta_, n_stars)
        alphafe_true = jnp.repeat(alphafe_, n_stars)
        meh_true = jnp.repeat(meh_, n_stars)
        ages_true = jnp.repeat(ages_norm_, n_stars)
        
        # Stack inputs for emulator
        x = jnp.stack([
            jnp.log10(massini_), 
            meh_true, 
            alphafe_true, 
            yini_true,
            jnp.log10(alphamlt_true), 
            eta_true, 
            ages_true
        ], axis=-1)
        
        # Call emulator
        y = call_emulator(input_norm=x, emulator=emulator, classical_outputs_len=len(classical_outputs))
        
        # Extract and transform physical parameters
        rad = numpyro.deterministic("rad", 10 ** y[..., 0])
        lum = numpyro.deterministic("luminosity", 10 ** y[..., 1])
        teff = numpyro.deterministic("teff", 5772 * ((rad ** (-2) * lum) ** (0.25)))
        
        # Compute asteroseismic parameters
        dnu = jnp.median(jnp.diff(y[..., len(classical_outputs):], axis=-1), axis=-1)
        numax = (dnu / 0.263) ** (1 / 0.772)  # Stello et al. relation
        
        # Observational likelihoods
        if obs is not None:
            numpyro.sample("teff_obs", dist.StudentT(5, teff, obs["teff_err"][0]), obs=obs["teff"])
            numpyro.sample("lum_obs", dist.StudentT(5, lum, obs["lum_err"][0]), obs=obs["lum"])
            numpyro.sample("dnu_obs", dist.StudentT(5, dnu, obs["dnu_err"][0]), obs=obs["dnu"])
            numpyro.sample("numax_obs", dist.StudentT(5, numax, obs["numax_err"][0]), obs=obs["numax"])
    
    return bayesian_model


def run_inference(model_func, observed_data, observed_errors, num_warmup=4000, num_samples=2000, num_chains=4, seed=0):
    """
    Run MCMC inference on the Bayesian model.
    
    Args:
        model_func: Bayesian model function
        observed_data: Dict of observed data arrays
        observed_errors: Dict of observed data errors
        num_warmup: Number of warmup steps
        num_samples: Number of samples to collect
        num_chains: Number of MCMC chains
        seed: Random seed
        
    Returns:
        ArviZ InferenceData object with posterior samples
    """
    # Combine data and errors
    obs = {
        "teff": observed_data["teff"],
        "lum": observed_data["lum"],
        "dnu": observed_data["dnu"],
        "numax": observed_data["numax"],
        "teff_err": observed_errors["teff_err"],
        "lum_err": observed_errors["lum_err"],
        "dnu_err": observed_errors["dnu_err"],
        "numax_err": observed_errors["numax_err"],
    }
    
    # Set up NUTS sampler and MCMC
    nuts = NUTS(
        model_func, 
        target_accept_prob=0.8, 
        init_strategy=init_to_median, 
        find_heuristic_step_size=True
    )
    
    mcmc = MCMC(
        nuts, 
        num_warmup=num_warmup, 
        num_samples=num_samples, 
        num_chains=num_chains
    )
    
    # Initialize random key
    rng = random.PRNGKey(seed)
    rng, key = random.split(rng)
    
    # Run inference
    mcmc.run(key, obs=obs)
    
    # Convert to ArviZ format
    trace = az.from_numpyro(mcmc)
    
    return trace


def main():
    """Main function to run the analysis."""
    # Load data
    df = pd.read_csv('HBM/tailo2022table.csv') 
    df_first_30 = df[:N_STARS]
    
    # Extract observed data
    Teff_M4 = jnp.array(df_first_30["Teff"])
    L_M4 = jnp.array(df_first_30["L"])
    errL_M4 = jnp.array(df_first_30["errL"])
    numax_M4 = jnp.array(df_first_30["numax"])
    errnumax_M4 = jnp.array(df_first_30["errnumax"])
    Deltanu_M4 = jnp.array(df_first_30["Deltanu"])
    errDeltanu_M4 = jnp.array(df_first_30["errDeltanu"])
    
    # File paths
    run_name = "smiley_stingray"
    logfile = f"HBM/smiley_stingray/log_{run_name}.json"
    pcafile = f"HBM/smiley_stingray/pca_{run_name}.json"
    historyfile = f"HBM/smiley_stingray/history_{run_name}.json"
    checkpointfile = f"HBM/smiley_stingray/{run_name}_checkpoint.h5"
    
    # Load run configuration
    with open(logfile, "r") as fp:
        config = json.load(fp)
        
        gridpath = config["gridpath"]
        gridfile = config["gridfile"]
        grid = os.path.join(gridpath, gridfile)
        
        seed = config["seed"]
        n_components = config["n_components"]
        
        batch_size_exp = config["batch_size_exp"]
        epochs = config["epochs"]
        test_size = config["test_size"]
        fractrain = config["fractrain"]
        
        inputs = config["inputs"]
        classical_outputs = config["classical_outputs"]
        nmin = config["nmin"]
        nmax = config["nmax"]
    
    # Define asteroseismic outputs
    astero_outputs = [f"nu_0_{i+1}" for i in range(nmin - 1, nmax)]
    outputs = classical_outputs + astero_outputs
    
    # Load emulator
    emulator = load_emulator(checkpointfile, pcafile)
    
    # Prepare observed data
    observed_data = {
        'teff': Teff_M4,
        'lum': L_M4,
        'dnu': Deltanu_M4,
        'numax': numax_M4
    }
    
    observed_errors = {
        'teff_err': jnp.full_like(Teff_M4, 70),
        'lum_err': errL_M4,
        'dnu_err': errDeltanu_M4,
        'numax_err': errnumax_M4
    }
    
    # Setup Bayesian model
    model = setup_bayesian_model(emulator, N_STARS, classical_outputs)
    
    # Run inference
    trace = run_inference(
        model, 
        observed_data, 
        observed_errors, 
        num_warmup=4000, 
        num_samples=2000,
        num_chains=NUM_CHAINS,
        seed=RANDOM_SEED
    )
    
    # Save results
    trace.to_netcdf("Bayesian_model_BBtest.nc")


if __name__ == "__main__":
    main()