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



def parse_args():
    "Parses inputs from commandline and returns them as a Namespace object."

    parser = ArgumentParser(prog = 'python3 tumor_growth.py',
        formatter_class = RawTextHelpFormatter, description =
        '  Simulate tumor growth given arguments for growth probability, grid size and number of time steps.\n\n'
        '  Example syntax:\n'
        '    python3 tumor_growth.py -N 50 -T 100 -p 0.1\n')

    # Optionals
    parser.add_argument('-N', dest='GRID_SIZE', type=int, default=50,
        help='grid size (default = 50)')

    parser.add_argument('-T', dest='TIME_STEPS', type=int, default=150,
        help='number of time steps (default = 100)')

    parser.add_argument('-p', dest='GROWTH_PROBABILITY', type=list, default= .3 , 
        help='growth probability, probability that tumor cell divides, (default = 0.3) ')

    parser.add_argument('-d', dest='DEATH_PROBABILITY', type=list, default=.01,
                        help='death probability, probability that tumor cell dies, (default = 0.01) ')
    
    parser.add_argument('-m', dest = 'MUTATION_PROBABILITY', type = float, default =   0  ,
                        help = 'mutation probability of a healthy cell into a cancer cell, (default = .00001)')
    return parser.parse_args()

def get_neighbors(x, y, N):
    """
    Get the neighboring coordinates of a given cell in an N x N grid.
    This function returns a list of valid neighboring coordinates (excluding the 
    cell itself) for a cell located at (x, y) in an N x N grid. The neighbors 
    are shuffled randomly before being returned.

    Parameters:
    x (int): The x-coordinate of the cell.
    y (int): The y-coordinate of the cell.
    N (int): The size of the grid (N x N).

    Returns:
    list of tuple: A list of tuples representing the coordinates of the neighboring cells.
    """
    neighbors = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if not (i == 0 and j == 0):
                nx, ny = x + i, y + j
                if 0 <= nx < N and 0 <= ny < N:
                    neighbors.append((nx, ny))
    np.random.shuffle(neighbors)
    assert isinstance(neighbors, list) and all(isinstance(coord, tuple) and len(coord) == 2 for coord in neighbors), "Neighbors must be a list of coordinate tuples."
    
    return neighbors

def mutate(p_mutate): 
    """
    Determines whether a mutation occurs based on a given probability.
    Parameters:
    p_mutate (float): The probability of mutation, a value between 0 and 1.
    Returns:
    int: Returns 1 if a mutation occurs (random number is less than p_mutate), otherwise returns 0.
    """

    return 1 if np.random.rand() < p_mutate else 0

def save_frame(grid, step):
    """
    Save a frame of the tumor growth simulation as an image file.
    Parameters:
    grid (ndarray): A 2D numpy array representing the current state of the grid.
    step (int): The current step number of the simulation.
    Returns:
    None
    """
    # Initialize figure 
    plt.figure(figsize=(5, 5))
    plt.imshow(grid, cmap='viridis', interpolation='nearest', vmin=0, vmax=2)
    plt.axis('off')
    plt.title(f"Tumor Growth at Step {step}")
    # Save the current frame to an image object
    # Create directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f'frame_{step}.png'), bbox_inches='tight', pad_inches=0)
    plt.close()

def save_gif(T):
    """
    Creates and saves a GIF from a series of saved frames.
    Parameters:
    T (int): The total number of frames to consider. Frames are selected at intervals of 5.
    The function assumes that the frames are saved as 'frame_{step}.png' in a 'data' directory
    within the current working directory. The resulting GIF is saved as 'tumor_growth_simulation.gif'
    in the same directory.
    The function also prints the path to the saved GIF for convenience.
    """

    # Create GIF from saved frames
    output_dir = os.path.join(os.getcwd(), 'data')
    images = [Image.open(os.path.join(output_dir, f'frame_{step}.png')) for step in range(0, T, 5)]
    gif_path = os.path.join(output_dir, 'tumor_growth_simulation.gif')
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=300, loop=0)

    # Display the path to download the GIF
    print(f"GIF saved as: {gif_path}")

def check_perc_cluster(grid):
    """
    Checks if there is a percolating cluster in the grid and returns its size.
    """
    # Label connected tumor cell clusters
    labeled_grid, num_features = label(grid == 1)  # Only tumor cells
    
    # Check if any cluster spans from top to bottom or left to right
    for cluster_id in range(1, num_features + 1):
        cells = np.argwhere(labeled_grid == cluster_id)
        rows, cols = cells[:, 0], cells[:, 1]
        if (0 in rows and grid.shape[0] - 1 in rows) or (0 in cols and grid.shape[1] - 1 in cols):
            print(f"Percolation found: Cluster ID {cluster_id}, Size {len(cells)}")

            cluster_size = len(cells)
            return True, cluster_size  # Percolating cluster found
    assert not any((0 in rows and grid.shape[0] - 1 in rows) or (0 in cols and grid.shape[1] - 1 in cols) for cluster_id in range(1, num_features + 1)), "No percolating cluster should be found."
    return False, 0  # No percolation


def find_largest_and_second_largest_components(grid):
    """
    Finds the largest and second largest connected components of tumor cells in the grid.
    """
    labeled_grid, num_features = label(grid == 1)  # Only tumor cells
    component_sizes = []

    for component_id in range(1, num_features + 1):
        component_size = np.sum(labeled_grid == component_id)
        component_sizes.append((component_id, component_size))

    # Sort components by size in descending order
    component_sizes.sort(key=lambda x: x[1], reverse=True)

    largest_component_id = component_sizes[0][0] if component_sizes else None
    largest_component_size = component_sizes[0][1] if component_sizes else 0

    second_largest_component_id = component_sizes[1][0] if len(component_sizes) > 1 else None
    second_largest_component_size = component_sizes[1][1] if len(component_sizes) > 1 else 0
    assert largest_component_size >= second_largest_component_size, "Largest component size should be greater than or equal to the second largest component size."

    return (largest_component_id, largest_component_size), (second_largest_component_id, second_largest_component_size)

def simulate_growth(N, T, p, d, m, save_plots = False):
    """
    Simulates the growth and death dynamics on a grid over a specified number of time steps.
    Parameters:
    N (int): The size of the grid (NxN).
    T (int): The number of time steps to simulate.
    p (list of float): List of growth probabilities.
    d (list of float): List of death probabilities.
    m (float): Mutation probability.
    save_plots (bool, optional): If True, saves the plots of the grid at each time step. Default is False.
    Returns:
    dict: A dictionary where keys are the ratio of growth to death probabilities and values are lists of tuples.
            Each tuple contains (run, step, size) where:
            - run (int): The run number.
            - step (int): The step at which percolation occurred.
            - size (int): The size of the percolating cluster."""
    
    

    k = 100   # Number of runs


    # Map of neighbours for all cells 
    neighbors_map = {
        (x, y): get_neighbors(x, y, N)
        for x in range(N) for y in range(N)
    }
    
    gc_portion = [[] for _ in range(k)]
    second_portion = [[] for _ in range(k)]
    
    for run in range(k):
        old_grid = np.zeros((N, N), dtype=int)
        center = N // 2
        old_grid[center, center] = 1 


                   
        for step in tqdm(range(T), desc=f"Run {run+1}/{k}"):
            new_grid = old_grid.copy()
                        

            for x, y in np.random.permutation([(i, j) for i in range(N) for j in range(N)]):
                assert new_grid[x, y] in [0, 1, 2], f"Unexpected cell value {new_grid[x, y]} at ({x}, {y})"

                if old_grid[x, y] == 1:                                            # If I am a tumor cell
                    
                    for nx, ny in neighbors_map[(x, y)]:
                        # If neighbors are healthy
                        if old_grid[nx, ny] == 0 and np.random.rand() < p:
                            new_grid[nx, ny] = 1                                    #Infect each neighbor with growth probability
                
                if old_grid[x, y] == 1 and np.random.rand() < d:
                    # this tumor cell dies with d probability
                    new_grid[x, y] = 2
                
                elif old_grid[x,y] == 0 : 
                    new_grid[x,y] = mutate(m)
            

            largest, second_largest = find_largest_and_second_largest_components(new_grid)
            gc_portion[run].append(largest[1])
            second_portion[run].append(second_largest[1])
            old_grid = new_grid
            
    return gc_portion, second_portion
       
def plot_growth(gc_portion, second_portion, k):
    """
    Plots the growth of the largest and second largest components over time.
    Parameters:
    gc_portion (list of lists): A list where each element is a list representing the portion of the grid 
                                occupied by the largest component at each time step for a single run.
    second_portion (list of lists): A list where each element is a list representing the portion of the grid 
                                    occupied by the second largest component at each time step for a single run.
    k (int): The number of runs or iterations.
    Returns:
    None: The function saves the plot as a PNG file in the 'data' directory.
    """
    # Plot the portion of the grid occupied by the largest component over time
    plt.figure(figsize=(10, 6))
    for i in range(len(gc_portion)): 
        plt.plot(gc_portion[i], alpha=0.05, color="orange")
        plt.plot(second_portion[i],alpha=0.05, color="blue")
    
    # Calculate and plot averages
    avg_gc_portion = np.mean(gc_portion, axis=0)
    avg_second_portion = np.mean(second_portion, axis=0)
    plt.plot(avg_gc_portion, label="Average Largest component", color="orange", linewidth=2)

    plt.plot(avg_second_portion, label="Average Second component", color="blue", linewidth=2)
    
    plt.xlabel('Time Steps')
    plt.ylabel('Component Size (cells)')
    plt.title(f'Growth of the Largest Tumor Component Over {k} steps (No Mutation)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'growth_largest_component_{k}_runs_nomut.png')
    plt.close()

def plot_percolation(ratio_dict, p ):
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
        color = 'gray' if avg_time == T else cmap(0.4 + 0.6 * norm(avg_time))  # Scale color to avoid very light shades

        # Plot the point
        if avg_size == 0:
            plt.scatter(ratio, avg_size, color=color, label=str(ratio), s=50, alpha=0.8)
        else:
            plt.scatter(ratio, avg_size, color=color, s=50, alpha=0.8)


    # Set log scale for axes
    plt.xscale('log')
    plt.legend()
    # Add labels, title, and grid
    plt.ylabel('Average Size of Perlocating Cluster', fontsize=14)
    plt.xlabel('mutation/Death Ratio', fontsize=14)
    plt.title(f' Average Size of perlocating cluster vs. mutation/Death Ratio \n growth  probability = {p}', fontsize=16)
    plt.grid(True)
    plt.savefig("Ratio_Percolating_size_MUTATION.png")
    # Show the plot
    plt.show()
    plt.close()


if __name__ == '__main__':
    args = parse_args()
    # Parameters
    N = args.GRID_SIZE  # Size of the grid (default 50x50)
    T = args.TIME_STEPS  # Number of simulation steps
    p = args.GROWTH_PROBABILITY  # Probability of tumor cell division
    d = args.DEATH_PROBABILITY # Probability that a tumor cell dies
    m = args.MUTATION_PROBABILITY # probabilty that a healthy cell turns into a tumor cell 
    


    
    largest, second = simulate_growth(N, T, p, d, m, save_plots = True)
    # (not saving to pickle file since runtime is short)

    #plot the graph 
    plot_growth(largest, second, 100)
    

