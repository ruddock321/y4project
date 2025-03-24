import os
import h5py
import numpy as np
import random
import joblib  # For saving scalers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define file paths
garstec_data = r'C:\Users\kiena\Documents\YEAR 4\PROJECT\Data\Garstec_AS09_chiara.hdf5'
save_dir = r'C:\Users\kiena\Python Project\Year4Project\HBM\BayesianModel'
os.makedirs(save_dir, exist_ok=True)

# 7 Inputs
ages = []
massini = []
fehini = []
alphamlt = []
yini = []
eta = []
alphafe = []

# 1 Output (only radius)
radius = []

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

        # Output (only radius)
        radius.append(track['radPhot'][:])

# Convert lists to numpy arrays and concatenate 
# Define a small constant to avoid log10(0)
epsilon = 1e-10

# Features requiring log10 transformation
log10_vars_inputs = [ages, yini]  # Removed massini, alphamlt, and eta

# Transform log10 variables
log10_transformed_inputs = [np.log10(np.maximum(np.concatenate(var).reshape(-1, 1), epsilon)) for var in log10_vars_inputs]

# Concatenate all inputs
inputs = np.hstack(log10_transformed_inputs + [
    np.concatenate(massini).reshape(-1, 1),       # Raw massini
    np.concatenate(alphamlt).reshape(-1, 1),      # Raw alphamlt
    np.concatenate(eta).reshape(-1, 1),           # Raw eta
    np.concatenate(fehini).reshape(-1, 1),
    np.concatenate(alphafe).reshape(-1, 1)
])

# Output - radius (no log10 transformation)
outputs = np.concatenate(radius).reshape(-1, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=1)

# Initialize and fit scalers
input_scaler = StandardScaler().fit(X_train)
output_scaler = StandardScaler().fit(y_train)

# Save scalers
joblib.dump(input_scaler, os.path.join(save_dir, 'input_scalerV14.pkl'))
joblib.dump(output_scaler, os.path.join(save_dir, 'output_scalerV14.pkl'))

print("Scalers saved successfully!")
