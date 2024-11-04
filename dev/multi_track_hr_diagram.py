import h5py
import matplotlib.pyplot as plt
import numpy as np

garstec_data = r'C:\Users\kiena\Documents\YEAR 4\PROJECT\Data\Garstec_AS09_chiara.hdf5'


# Open the hdf5 file (read-only mode)
with h5py.File(garstec_data, 'r') as hdf:

    '''
    We want to make a HR diagram plot with a sample of mass (1.0 < M < 1.1) 
    solar masses, and alpha_mlt initially set to ~1.79.
    '''

    # Navigate through Grid -> tracks -> track00001
    grid = hdf['grid']
    tracks = grid['tracks']

    # Initialize a list to hold the tracks that meet our desired criteria
    selected_tracks = []

    for track_name in tracks:

        track = tracks[track_name]

        # Check if 'massini' dataset exists and is not empty
        if 'massini' in track and track['massini'].size > 0:
            massini_value = track['massini'][()][0]  # Get the first element if it exists
        else:
            massini_value = 0  # Put outside of our desired range so it isn't selected

        # Check if 'alphaMLT' dataset exists and is not empty
        if 'alphaMLT' in track and track['alphaMLT'].size > 0:
            alphaMLT_value = track['alphaMLT'][()][0]  # Get the first element if it exists
        else:
            alphaMLT_value = 0  # Put outside of our desired range so it isn't selected

        if (1.0 < massini_value < 1.1) and (1.785 < alphaMLT_value < 1.795): 
            selected_tracks.append(track)
            

    # Now for the fun bit, plot the HR diagrams for each of the selected tracks

    for track in selected_tracks:
        # Access effective temperature and luminosity as np arrays
        teff_data = track['Teff'][:] 
        luminosity_data = track['LPhot'][:] 

        track_name = track.name.rsplit('/', 1)[-1]  # Split from the right, keep the last part

        # Plot effective temp against luminosity
        plt.scatter(teff_data, luminosity_data, s=1, label=f'{track_name}')


    plt.gca().invert_xaxis() #Invert x-axis to have proper form of HR diagram
    plt.xlabel("Effective Temperature (K)")
    plt.ylabel("Luminosity ($L_\odot$)")
    plt.title("Effective Temperature vs. Luminosity")
    plt.legend()
    plt.show()