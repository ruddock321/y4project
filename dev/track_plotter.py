import h5py
import matplotlib.pyplot as plt
import numpy as np

garstec_data = r'C:\Users\kiena\Documents\YEAR 4\PROJECT\Data\Garstec_AS09_chiara.hdf5'

# Open the hdf5 file (read-only mode)
with h5py.File(garstec_data, 'r') as hdf:

    # Navigate through Grid -> tracks -> track00001
    grid = hdf['grid']
    tracks = grid['tracks']
    track00001 = tracks['track00001']

    # Access effective temperature and luminosity as np arrays
    teff_data = track00001['Teff'][:] # Reverses array for correct HR diagram
    luminosity_data = track00001['LPhot'][:] 

    # Plot effective temp against luminosity
    plt.scatter(teff_data, luminosity_data, s=3, label='Effective Temperature vs. Luminosity')
    plt.gca().invert_xaxis() #Invert x-axis to have proper form of HR diagram
    plt.xlabel("Effective Temperature (K)")
    plt.ylabel("Luminosity ($L_\odot$)")
    plt.title("Effective Temperature vs. Luminosity")
    plt.show()
