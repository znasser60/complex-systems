import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy.ndimage import laplace
from argparse import ArgumentParser, RawTextHelpFormatter
from scipy.ndimage import label

def parse_args():
    """Parses command-line arguments."""
    parser = ArgumentParser(
        prog='python3 tumor_growth.py',
        formatter_class=RawTextHelpFormatter,
        description=(
            'Simulate tumor growth given arguments for growth probability, grid size, and number of time steps.\n\n'
            'Example syntax:\n'
            '  python3 tumor_growth.py -N 50 -T 100 -p 0.1\n'
        )
    )

    parser.add_argument('-N', dest='GRID_SIZE', type=int, default=100, help='Grid size (default: 100)')
    parser.add_argument('-T', dest='TIME_STEPS', type=int, default=100, help='Number of time steps (default: 100)')
    parser.add_argument('-p', dest='GROWTH_PROBABILITY', type=float, default=0.6, help='Tumor cell division probability (default: 0.6)')
    parser.add_argument('-od', dest='OXYGEN_DIFFUSION', type=float, default=2.41e-5, help='Oxygen diffusion coefficient (default: 2.41e-5)')
    parser.add_argument('-cu', dest='CANCER_UPTAKE', type=float, default=1.57e-4, help='Cancer cell oxygen uptake rate (default: 1.57e-4)')
    parser.add_argument('-nu', dest='NORMAL_UPTAKE', type=float, default=1.57e-4, help='Normal cell oxygen uptake rate (default: 1.57e-4)')
    parser.add_argument('-o', dest='INITIAL_OXYGEN', type=float, default=8.5e-3, help='Initial oxygen concentration (default: 8.5e-3)')
    parser.add_argument('-nt1', dest='NT1', type=float, default=4.5e-4, help='Oxygen threshold for normal cells with normal neighbors (default: 4.5e-4)')
    parser.add_argument('-nt2', dest='NT2', type=float, default=4.5e-3, help='Oxygen threshold for normal cells with cancer neighbors (default: 4.5e-3)')
    parser.add_argument('-ct1', dest='CT1', type=float, default=1.5e-5, help='Oxygen threshold for cancer cells with cancer neighbors (default: 1.5e-5)')
    parser.add_argument('-ct2', dest='CT2', type=float, default=4.5e-5, help='Oxygen threshold for cancer cells with normal neighbors (default: 4.5e-5)')
    parser.add_argument('-qt', dest='QUIESCENCE_TIME', type=int, default=20, help='Quiescence time for cancer cells (default: 20)')
    parser.add_argument('--output-path', dest='OUTPUT_PATH', type=str, default='./', help='Path to save frames and GIF')
    return parser.parse_args()

def initialize_grid(N): 
    '''
    Initializes cellular automata grid with two 5 x 5 clumps of cells where 50% of the clumps are normal cells and the remaining are cancer cells. 
    Initializes oxygen grid with homogeneous oxygen levels equal to the set initial oxygen level. 
    '''
    cell_grid = np.zeros((N, N), dtype=int)
    oxygen_grid = np.full((N, N), args.INITIAL_OXYGEN)

    clump_centers = [(int(N * 0.25), int(N * 0.25)), (int(N * 0.75), int(N * 0.75))]
    for center in clump_centers:
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                x, y = center[0] + dx, center[1] + dy
                if 0 <= x < N and 0 <= y < N:
                    cell_grid[x, y] = 1 if np.random.rand() < 0.5 else 2

    return cell_grid, oxygen_grid

def get_neighbors(x, y, N): 
    '''
    Gets the neighbor of each specified cell using Moore's neighborhood. 
    '''
    neighbors = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if not (i == 0 and j == 0):
                nx, ny = x + i, y + j
                if 0 <= nx < N and 0 <= ny < N:
                    neighbors.append((nx, ny))
    return neighbors

def count_neighbors(x, y, cell_grid, N):
    '''
    Counts the number of normal and cancer cells neighboring a specified cell.
    '''
    neighbors = get_neighbors(x, y, N)
    normal = sum(1 for nx, ny in neighbors if cell_grid[nx, ny] == 1)
    cancer = sum(1 for nx, ny in neighbors if cell_grid[nx, ny] == 2)
    return normal, cancer

def update_oxygen(cell_grid, oxygen_grid, normal_uptake, cancer_uptake, diffusion):
    '''
    Updates the oxygen grid based on an oxygen diffusion equation. 
    '''
    uptake = np.zeros_like(oxygen_grid)
    uptake[cell_grid == 1] = normal_uptake
    uptake[cell_grid == 2] = cancer_uptake
    return np.clip(oxygen_grid + diffusion * laplace(oxygen_grid) - uptake, 0, 1)

def simulate_growth(args):
    '''
    Runs the tumor growth simulation based on stated cellular automata rules.
    '''
    N = args.GRID_SIZE
    T = args.TIME_STEPS

    cell_grid, oxygen_grid = initialize_grid(N)
    quiescent_clock = np.zeros((N, N), dtype=int)
    
    for step in range(T):
        new_grid = cell_grid.copy()
        for x in range(N):
            for y in range(N):
                if cell_grid[x, y] == 1:  # Normal cell
                    normal_cell_neighbors, cancer_cell_neighbors = count_neighbors(x, y, cell_grid, N)
                    local_oxygen_threshold = args.NT1 if normal_cell_neighbors > cancer_cell_neighbors else args.NT2
                    if oxygen_grid[x, y] < local_oxygen_threshold:  
                        new_grid[x, y] = 0
                    else:  # Attempts to divide
                        neighbors = get_neighbors(x, y, N)
                        empty_neighbors = [(nx, ny) for nx, ny in neighbors if cell_grid[nx, ny] == 0]
                        if empty_neighbors and np.random.rand() < args.GROWTH_PROBABILITY:
                            nx, ny = empty_neighbors[np.random.choice(len(empty_neighbors))]
                            new_grid[nx, ny] = 1

                elif cell_grid[x, y] == 2:  # Cancer cell
                    normal_cell_neighbors, cancer_cell_neighbors = count_neighbors(x, y, cell_grid, N)
                    local_oxygen_threshold = args.CT1 if cancer_cell_neighbors > normal_cell_neighbors else args.CT2
                    if oxygen_grid[x, y] < local_oxygen_threshold: 
                        new_grid[x, y] = 3
                    else:  # Attempts to divide
                        neighbors = get_neighbors(x, y, N)
                        empty_neighbors = [(nx, ny) for nx, ny in neighbors if cell_grid[nx, ny] == 0]
                        if empty_neighbors and np.random.rand() < args.GROWTH_PROBABILITY:
                            nx, ny = empty_neighbors[np.random.choice(len(empty_neighbors))]
                            new_grid[nx, ny] = 2

                elif cell_grid[x, y] == 3:  # Quiescent cancer cell
                    normal_cell_neighbors, cancer_cell_neighbors = count_neighbors(x, y, cell_grid, N)
                    local_oxygen_threshold = args.CT1 if cancer_cell_neighbors > normal_cell_neighbors else args.CT2
                    if oxygen_grid[x, y] > local_oxygen_threshold: 
                        new_grid[x, y] = 2
                        quiescent_clock[x, y] = 0
                    else:
                        quiescent_clock[x, y] += 1
                        if quiescent_clock[x, y] > args.QUIESCENCE_TIME:  
                            new_grid[x, y] = 0

        cell_grid = new_grid
        oxygen_grid = update_oxygen(cell_grid, oxygen_grid, args.NORMAL_UPTAKE, args.CANCER_UPTAKE, args.OXYGEN_DIFFUSION)

    final_cancer_cells = np.sum(cell_grid == 2) + np.sum(cell_grid == 3)
    return cell_grid, final_cancer_cells

def percolation_analysis(cell_grid):
    """
    Perform percolation analysis on the tumor grid to identify cancer cell clusters.
    """
    cancer_cells = (cell_grid == 2)
    labeled_grid, num_clusters = label(cancer_cells)
    cluster_sizes = np.bincount(labeled_grid.ravel())[1:]  # Exclude background cluster (label 0)
    spans_top_to_bottom = any(
        np.intersect1d(np.where(labeled_grid[0, :] > 0)[0], np.where(labeled_grid[-1, :] > 0)[0])
    )
    spans_left_to_right = any(
        np.intersect1d(np.where(labeled_grid[:, 0] > 0)[0], np.where(labeled_grid[:, -1] > 0)[0])
    )

    percolates = spans_top_to_bottom or spans_left_to_right
    
    return num_clusters, cluster_sizes, percolates

def plot_percolation_threshold(args, oxygen_levels, num_simulations=1):
    """
    Plot oxygen levels against the size of the largest cluster to identify the percolation threshold.
    """
    largest_cluster_sizes = []

    for oxygen in oxygen_levels:
        args.INITIAL_OXYGEN = oxygen  # Set the current oxygen level
        largest_sizes_for_simulation = []

        for _ in range(num_simulations):
            cell_grid, _ = simulate_growth(args)
            _, cluster_sizes, percolates = percolation_analysis(cell_grid)

            if len(cluster_sizes) > 0:
                largest_sizes_for_simulation.append(max(cluster_sizes))
            else:
                largest_sizes_for_simulation.append(0)

        avg_largest_cluster = np.mean(largest_sizes_for_simulation)
        largest_cluster_sizes.append(avg_largest_cluster)

    plt.figure(figsize=(10, 5))
    plt.plot(oxygen_levels, largest_cluster_sizes, marker='o', color='blue', label='Largest Cluster Size')
    percolation_index = np.argmax(np.diff(largest_cluster_sizes))
    percolation_oxygen = oxygen_levels[percolation_index]
    plt.axvline(x=percolation_oxygen, color='red', linestyle='--', label=f'Percolation Threshold: {percolation_oxygen:.5f}')
    plt.title("Percolation Threshold: Largest Cluster Size vs. Oxygen Level")
    plt.xlabel("Oxygen Level")
    plt.ylabel("Largest Cluster Size")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    simulate_growth(args)

    oxygen_levels = oxygen_levels = np.linspace(1e-5, 5e-4, 20)
    plot_percolation_threshold(args, oxygen_levels, num_simulations=1)