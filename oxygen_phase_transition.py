import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy.ndimage import laplace
from argparse import ArgumentParser, RawTextHelpFormatter
from tumor_growth_with_oxygen import parse_args, initialize_grid, get_neighbors, count_neighbors, update_oxygen, simulate_growth

def run_oxygen_simulations(args, oxygen_levels, num_simulations=5):
    '''
    Runs tumor growth simulations for different oxygen levels with multiple simulations.
    Saves results to a file for quick re-plotting.
    '''

    all_simulation_results = {oxygen: [] for oxygen in oxygen_levels}
    avg_counts_all_simulations = []

    plt.figure(figsize=(10, 5))

    for sim in range(num_simulations): 
        final_counts_for_simulation = []  
        for oxygen in oxygen_levels:
            args.INITIAL_OXYGEN = oxygen  
            _, final_cancer_cells = simulate_growth(args)  
            final_counts_for_simulation.append(final_cancer_cells) 

            all_simulation_results[oxygen].append(final_cancer_cells)

        plt.plot(oxygen_levels, final_counts_for_simulation, color="red", alpha=0.3)

    for oxygen in oxygen_levels:
        avg_count = np.mean(all_simulation_results[oxygen]) 
        avg_counts_all_simulations.append(avg_count)

    plt.plot(oxygen_levels, avg_counts_all_simulations, color="red")
    transition_index = np.argmax(np.diff(avg_counts_all_simulations))  # Phase transition line (biggest change)
    plt.axvline(x=oxygen_levels[transition_index], color='black', linestyle='--', label=f'Phase Transition: Oxygen Level {oxygen_levels[transition_index]:.7f}')
    plt.title("Tumor Growth Simulations Across Varying Oxygen Levels")
    plt.xlabel("Initial Oxygen Level")
    plt.ylabel("Final Cancer Cell Count")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    oxygen_levels = np.linspace(1e-5, 5e-4, 20)
    run_oxygen_simulations(args, oxygen_levels, num_simulations=5)