import numpy as np
import h5py
import random
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Timer
from pytorch_lightning.tuner.tuning import Tuner

# Set PyTorch to use full precision by default
torch.set_default_dtype(torch.float32)

timer_callback = Timer(duration="00:48:00:00")  # dd:hh:mm:ss

# Print Python, PyTorch, and CUDA version information
print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Default tensor type:", torch.get_default_dtype())

# Check the number of GPUs available
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")

# If GPUs are available, print their names
if torch.cuda.is_available():
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPUs detected. Using CPU.")

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# Check the number of CPU cores
num_cores = os.cpu_count()
print("Number of cores: ", num_cores)


# RDS directory
garstec_data = r'/rds/projects/d/daviesgr-m4massloss/Garstec_AS09_chiara.hdf5'
save_dir = r'/rds/projects/d/daviesgr-m4massloss/GarstecNN_V11'
os.makedirs(save_dir, exist_ok=True)

# Path to the checkpoint
checkpoint_path = r'best_model_v11_SS---epoch=8338-val_loss=67.00212860.ckpt'

# 7 Inputs
ages = []
massini = []
fehini = []
alphamlt = []
yini = []
eta = []
alphafe = []

# 5 Outputs (removed massfin, G_GAIA, and MeH)
teff = []
luminosity = []
dnufit = []
FeH = []
numax = []

# Open the hdf5 file (read-only mode)
with h5py.File(garstec_data, 'r') as hdf:
    grid = hdf['grid']
    tracks = grid['tracks']

    # Get a list of track names and shuffle for random sampling
    track_names = list(tracks.keys())
    random.seed(1)
    random.shuffle(track_names)

    # Choose a subset of tracks to process (or not)
    selected_tracks = track_names[:]

    for track_name in selected_tracks:  # Iterate over the selected track names
        track = tracks[track_name]
        # Inputs
        ages.append(track['age'][:])
        massini.append(track['massini'][:])
        fehini.append(track['FeHini'][:])
        alphamlt.append(track['alphaMLT'][:])
        yini.append(track['yini'][:])
        eta.append(track['eta'][:])
        alphafe.append(track['alphaFe'][:])

        # Outputs (removed massfin, G_GAIA, and MeH)
        teff.append(track['Teff'][:])
        luminosity.append(track['LPhot'][:])
        dnufit.append(track['dnufit'][:])
        FeH.append(track['FeH'][:])
        numax.append(track['numax'][:])

# Convert lists to numpy arrays and concatenate 
# Define a small constant to avoid log10(0)
epsilon = 1e-10

# Features requiring log10 transformation
log10_vars_inputs = [ages, massini, alphamlt, eta, yini]

# Transform log10 variables
log10_transformed_inputs = [np.log10(np.maximum(np.concatenate(var).reshape(-1, 1), epsilon)) for var in log10_vars_inputs]

# Concatenate all inputs, including raw `fehini` and `yini`
inputs = np.hstack(log10_transformed_inputs + [np.concatenate(fehini).reshape(-1, 1), 
                                             np.concatenate(alphafe).reshape(-1, 1)])

# Features requiring log10 transformation (strictly positive outputs)
log10_vars_outputs = [teff, luminosity, dnufit, numax]  # Removed massfin

# Transform log10 variables
log10_transformed_outputs = [np.log10(np.maximum(np.concatenate(var).reshape(-1, 1), epsilon)) for var in log10_vars_outputs]

# Combine transformed log10 outputs with raw FeH (removed MeH and G_GAIA)
outputs = np.hstack(log10_transformed_outputs + [np.concatenate(FeH).reshape(-1, 1)])

X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=1)

# Initialize scalers
input_scaler = StandardScaler()
output_scaler = StandardScaler()

# Fit scalers on training data and transform both training and testing data
X_train_scaled = input_scaler.fit_transform(X_train) # Fit AND transform train
X_test_scaled = input_scaler.transform(X_test)       # ONLY transform test

y_train_scaled = output_scaler.fit_transform(y_train)
y_test_scaled = output_scaler.transform(y_test)


# Data Module
class GarstecDataModule(LightningDataModule):
    def __init__(self, X_train, X_test, y_train, y_test, batch_size=2**18):
        super().__init__()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.batch_size = batch_size
        self.num_workers = min(16, os.cpu_count())

    def train_dataloader(self):
        train_dataset = TensorDataset(
            torch.tensor(self.X_train, dtype=torch.float32),
            torch.tensor(self.y_train, dtype=torch.float32)
        )
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    def val_dataloader(self):
        val_dataset = TensorDataset(
            torch.tensor(self.X_test, dtype=torch.float32),
            torch.tensor(self.y_test, dtype=torch.float32)
        )
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)

# Lightning Module
class GarstecNet(LightningModule):
    def __init__(self, input_dim, output_dim, loss_weights, lr=1e-3, 
                 output_scaler_mean=None, output_scaler_scale=None):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.loss_weights = torch.tensor(loss_weights)
        
        # Store scaler parameters as tensors
        self.register_buffer('output_mean', torch.tensor(output_scaler_mean, dtype=torch.float32))
        self.register_buffer('output_scale', torch.tensor(output_scaler_scale, dtype=torch.float32))

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, output_dim)
        )
    
    def inverse_transform(self, scaled_tensor):
        """PyTorch implementation of inverse_transform from scikit-learn's StandardScaler"""
        return scaled_tensor * self.output_scale + self.output_mean
    
    def custom_loss(self, y_pred, y_true):
        # Both y_pred and y_true are scaled tensors
        
        # Inverse transform predictions to get original scale
        y_pred_original = self.inverse_transform(y_pred)
        y_true_original = self.inverse_transform(y_true)
        
        # Convert log outputs back to linear scale for the first 4 columns
        # Create a mask for log-transformed features (first 4 columns)
        log_mask = torch.zeros_like(y_pred_original)
        log_mask[:, :4] = 1.0  # First 4 columns were log-transformed
        
        # Apply 10^x to log-transformed features
        normal_pred = torch.where(log_mask > 0, 
                                 10**y_pred_original, 
                                 y_pred_original)
        normal_true = torch.where(log_mask > 0, 
                                 10**y_true_original, 
                                 y_true_original)
        
        # Calculate weighted MSE loss
        weights = self.loss_weights.to(device=y_pred.device, dtype=y_pred.dtype)
        squared_diff = torch.pow(normal_true - normal_pred, 2)
        weighted_squared_diff = squared_diff / torch.pow(weights.view(1, -1), 2)
        
        # Average across all dimensions (both batch and features)
        loss = torch.mean(weighted_squared_diff)
        return loss
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.custom_loss(y_pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.custom_loss(y_pred, y)
        
        # Calculate MSE for each output feature separately for monitoring
        mse_per_feature = torch.mean(torch.pow(y - y_pred, 2), dim=0)
        feature_names = ['teff', 'luminosity', 'dnufit', 'numax', 'feh']
        
        # Log individual MSEs for monitoring
        for i, name in enumerate(feature_names):
            self.log(f'val_mse_{name}', mse_per_feature[i], on_epoch=True)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.9, patience=50, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            },
        }
    

# Define batch size and data module
batch_size = 2**18
data_module = GarstecDataModule(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, batch_size=batch_size)

# Define weights for [T_eff, L, dnu, numax, feh]
loss_weights = [70, 0.1, 0.1, 0.5/3090, 0.01] # Divide by 3090 to ensure correct units
print("Loss weights: ",loss_weights)

# Store the scaler parameters
output_scaler_mean = output_scaler.mean_
output_scaler_scale = output_scaler.scale_

# Define model and trainer
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]

# Load model from checkpoint - run the learning rate finder on the loaded model
model = GarstecNet.load_from_checkpoint(
    input_dim=input_dim, 
    output_dim=output_dim, 
    loss_weights=loss_weights, 
    lr=1e-3,
    output_scaler_mean=output_scaler_mean,
    output_scaler_scale=output_scaler_scale
)

print(f"Loaded model from checkpoint: {checkpoint_path}")
print(f"Current model learning rate: {model.lr}")

# Callbacks
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=save_dir,
    filename='continued_model_v11_SP_SS-AdamW-LeakyReLU---{epoch:02d}-{val_loss:.8f}',
    save_top_k=1,
    mode='min'
)

lr_monitor = LearningRateMonitor(logging_interval='epoch')

# Create the main trainer for continued training
trainer = Trainer(
    max_epochs=10000,  
    devices=1,
    accelerator='gpu',
    num_nodes=1,
    callbacks=[checkpoint_callback, timer_callback],
    log_every_n_steps=10,
    precision='32-true',  
)

# Create a Tuner
tuner = Tuner(trainer)

# Find optimal learning rate for fine-tuning
print("Running learning rate finder for fine-tuning...")
lr_finder = tuner.lr_find(model, datamodule=data_module)

new_lr = lr_finder.suggestion()
print(f"Suggested Learning Rate for fine-tuning: {new_lr}")

# Update the model's learning rate
model.lr = new_lr
model.hparams.lr = new_lr

# Configure the optimizers with the new learning rate and scheduler
model.configure_optimizers()


trainer.fit(model, data_module)

# Save the final model
final_checkpoint_path = checkpoint_callback.best_model_path
print(f"Final model saved to {final_checkpoint_path}")