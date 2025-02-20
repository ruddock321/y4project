import numpy as np
import h5py
import random
import os
import time
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Timer

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

start_time = time.time()
timer_callback = Timer(duration="00:44:00:00")  # dd:hh:mm:ss

# Print Python, PyTorch, and CUDA version information
print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

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

# Windows directory
#garstec_data = r'C:\Users\kiena\Documents\YEAR 4\PROJECT\Data\Garstec_AS09_chiara.hdf5'
#save_dir = r'C:\Users\kiena\Documents\YEAR 4\PROJECT\Data'
#os.makedirs(save_dir, exist_ok=True)

# RDS directory
garstec_data = r'/rds/projects/d/daviesgr-m4massloss/Garstec_AS09_chiara.hdf5'
save_dir = r'/rds/projects/d/daviesgr-m4massloss/GarstecNN_V8'
os.makedirs(save_dir, exist_ok=True)

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


# Data Module
class GarstecDataModule(LightningDataModule):
    def __init__(self, X_train, X_test, y_train, y_test, batch_size=2**17):
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
    def __init__(self, input_dim, output_dim, learning_rate=5e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=False)
        return loss

    def configure_optimizers(self):
        # Using SGD with momentum 
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0, 
            weight_decay=1e-5  # Adding small weight decay for regularization
        )
    
        # Modifying the learning rate scheduler for SGD
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,  # Reduce LR by half when plateau is detected
            patience=30,  # Wait for 20 epochs before reducing LR
            min_lr=1e-7,  # Minimum learning rate
            verbose=True
        )
    
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",  # Monitor train loss for scheduler
                "interval": "epoch",
                "frequency": 1
            }
        }
    
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=1)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Define batch size and data module
batch_size = 2**17
data_module = GarstecDataModule(X_train, X_test, y_train, y_test, batch_size=batch_size)

# Initialize model and load checkpoint
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
model = GarstecNet.load_from_checkpoint(
    checkpoint_path="fine_tuned_model_v8-epoch=11847-train_loss=0.00627.ckpt",
    input_dim=input_dim,
    output_dim=output_dim,
    learning_rate=5e-3
)

# New checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='train_loss',  # Changed to train_loss
    dirpath=save_dir,
    filename='fine_tuned_model_v8-{epoch:02d}-{train_loss:.5f}',  # More decimal places
    save_top_k=1,
    mode='min',
    every_n_epochs=1
)

lr_monitor = LearningRateMonitor(logging_interval='step')

trainer = Trainer(
    max_epochs=50000,
    devices=1,
    accelerator='gpu',
    num_nodes=1,
    precision="16-mixed",
    callbacks=[checkpoint_callback, lr_monitor, timer_callback],
    log_every_n_steps=50,
)

# Continue training
trainer.fit(model, data_module)