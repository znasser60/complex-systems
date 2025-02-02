import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from argparse import ArgumentParser, RawTextHelpFormatter
import os
from scipy.ndimage import label
from itertools import product 
from tqdm import tqdm
import pickle
import os



def plot_percolation(ratio_dict, p  ):
    """
    Plots the average size of percolating clusters against the growth/death ratio.
    Parameters:
    ratio_dict (dict): A dictionary where keys are growth/death ratios and values are lists of tuples.
                       Each tuple contains (time, size) of the percolating cluster for a specific run.
    The function creates a scatter plot where each point represents the average size of the percolating
    cluster for a given growth/death ratio. The color of the points is determined by the average time it took for the system to show percolation,
    normalized and mapped to the 'Blues' colormap. The plot is saved as "Ratio_Percolating_size.png" and displayed.
    Returns:
    None
    """  
    # Divide every ratio by 8
    ratio_dict = {key / 8: value for key, value in ratio_dict.items()}

    #Initialize figure 
    plt.figure(figsize=(10, 6))

    # Get all average sizes to normalize the colormap
    all_avg_sizes = [
        np.mean([run[2] for run in ratio_dict[key]]) 
        for key in ratio_dict if ratio_dict[key]
    ]
    
    all_avg_times = [
        np.mean([run[1] for run in ratio_dict[key]]) 
        for key in ratio_dict if ratio_dict[key]
    ]
    assert all_avg_times, "No data to plot."


    # Normalize using the min and max of average sizes
    norm = plt.Normalize(vmin=min(all_avg_times), vmax=max(all_avg_times))
    cmap = plt.cm.Blues  # Use the 'Blues' colormap
    
    # Iterate over each growth/death ratio in the dictionary
    for ratio, runs in ratio_dict.items():
        if not runs:
            continue  # Skip if no data for this ratio

        # Extract times and sizes for this ratio
        times = [run[1] for run in runs]
        sizes = [run[2] for run in runs]

        # Compute average time and size
        avg_time = np.mean(times)
        avg_size = np.mean(sizes)

        # Get the color based on the average size
        color = 'gray' if avg_time == 1000 else cmap(0.4 + 0.6 * norm(avg_time))  # Scale color to avoid very light shades

        plt.scatter(ratio, avg_size, color=color, s=50, alpha=0.8)


    # Set log scale for axes
    plt.axvline(1, ymin=0, linestyle='--', color = 'orange', label = "growth probability = death probability")
    plt.xscale('log')
    plt.legend()
    # Add labels, title, and grid
    plt.ylabel('Average Size of Perlocating Cluster', fontsize=14)
    plt.xlabel('Growth/Death Ratio', fontsize=14)
    plt.title(f' Average Size of perlocating cluster vs. Growth/Death Ratio \n mutation probability  probability = {p}', fontsize=16)
    plt.grid(True)
    plt.savefig("zzzzzzzzzzz.png")
    # Show the plot
    plt.show()
    plt.close()




    # Load the ratio_dict from the pickle file
with open(os.path.join('ratio_dict.pkl'), 'rb') as f:
    ratio_dict = pickle.load(f)
    
    #plot the graph 
plot_percolation(ratio_dict, .0001)