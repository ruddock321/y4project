import h5py

file = r'C:\Users\kiena\Documents\YEAR 4\PROJECT\Garstec_AS09_chiara.hdf5'

with h5py.File(file, 'r') as hdf:
    # List all groups in the file
    print("Keys in the file:", list(hdf.keys()))

with h5py.File(file, 'r') as hdf:
    grid = hdf['grid']
    if isinstance(grid, h5py.Dataset):
        print("The 'grid' is a dataset.")
    elif isinstance(grid, h5py.Group):
        print("The 'grid' is a group.")

with h5py.File(file, 'r') as hdf:
    grid_group = hdf['grid']
    print("Keys inside 'grid':", list(grid_group.keys()))

with h5py.File(file, 'r') as hdf:
    tracks = hdf['grid']['tracks']
    
    if isinstance(tracks, h5py.Dataset):
        print("'tracks' is a dataset.")
    elif isinstance(tracks, h5py.Group):
        print("'tracks' is a group.")

with h5py.File(file, 'r') as hdf:
    tracks_group = hdf['grid']['tracks']
    
    print("Keys inside 'tracks':", list(tracks_group.keys()))  # List datasets/subgroups
    
    # Check if the items inside 'tracks' are datasets or further groups
    for key in tracks_group.keys():
        item = tracks_group[key]
        if isinstance(item, h5py.Dataset):
            print(f"'{key}' inside 'tracks' is a dataset with shape: {item.shape}")
            print(f"First 5 entries of '{key}':", item[:5])  # Optionally, view the first few data points
        elif isinstance(item, h5py.Group):
            print(f"'{key}' inside 'tracks' is a group")


with h5py.File(file, 'r') as hdf:
    track_0_group = hdf['grid']['tracks']['track10000']  # Access the first track group (replace with actual track key)
    print("Keys inside 'track10000':", list(track_0_group.keys()))  # List the contents of the group
