import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy.ndimage import laplace
from argparse import ArgumentParser, RawTextHelpFormatter

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
    parser.add_argument('-T', dest='TIME_STEPS', type=int, default=150, help='Number of time steps (default: 150)')
    parser.add_argument('-p', dest='GROWTH_PROBABILITY', type=float, default=0.6, help='Tumor cell division probability (default: 0.6)')
    parser.add_argument('-od', dest='OXYGEN_DIFFUSION', type=float, default=2.41e-5, help='Oxygen diffusion coefficient (default: 2.41e-5)')
    parser.add_argument('-cu', dest='CANCER_UPTAKE', type=float, default=1.57e-4, help='Cancer cell oxygen uptake rate (default: 1.57e-4)')
    parser.add_argument('-nu', dest='NORMAL_UPTAKE', type=float, default=1.57e-4, help='Normal cell oxygen uptake rate (default: 1.57e-4)')
    parser.add_argument('-o', dest='INITIAL_OXYGEN', type=float, default=1.1e-2, help='Initial oxygen concentration (default: 1.1e-2)')
    parser.add_argument('-nt1', dest='NT1', type=float, default=4.5e-4, help='Oxygen threshold for normal cells with normal neighbors (default: 4.5e-4)')
    parser.add_argument('-nt2', dest='NT2', type=float, default=4.5e-3, help='Oxygen threshold for normal cells with cancer neighbors (default: 4.5e-3)')
    parser.add_argument('-ct1', dest='CT1', type=float, default=1.5e-5, help='Oxygen threshold for cancer cells with cancer neighbors (default: 1.5e-5)')
    parser.add_argument('-ct2', dest='CT2', type=float, default=4.5e-5, help='Oxygen threshold for cancer cells with normal neighbors (default: 4.5e-5)')
    parser.add_argument('-qt', dest='QUIESCENCE_TIME', type=int, default=75, help='Quiescence time for cancer cells (default: 75)')
    parser.add_argument('--output-path', dest='OUTPUT_PATH', type=str, default='./', help='Path to save frames and GIF')
    return parser.parse_args()

def initialize_grid(N, args): 
    '''
    Initializes cellular automata grid with two 5 x 5 clumps of cells where 50% of the clumps are normal cells and the remaining are cancer cells. 
    Initializes oxygen grid with homogeneous oxygen levels equal to the set initial oxygen level. 
    '''
    cell_grid = np.zeros((N, N), dtype=int)
    oxygen_grid = np.full((N, N), args.INITIAL_OXYGEN)

    clump1_center = (int(N * 0.5), int(N * 0.5))  
    for dx in range(-2, 3):  
        for dy in range(-2, 3):
            if (clump1_center[0] + dx >= 0 and clump1_center[0] + dx < N and
                clump1_center[1] + dy >= 0 and clump1_center[1] + dy < N):
                if np.random.rand() < 0.5:
                    cell_grid[clump1_center[0] + dx, clump1_center[1] + dy] = 2
                else:
                    cell_grid[clump1_center[0] + dx, clump1_center[1] + dy] = 1

    # clump_centers = [(int(N * 0.25), int(N * 0.25)), (int(N * 0.75), int(N * 0.75))]
    # for center in clump_centers:
    #     for dx in range(-2, 3):
    #         for dy in range(-2, 3):
    #             x, y = center[0] + dx, center[1] + dy
    #             if 0 <= x < N and 0 <= y < N:
    #                 cell_grid[x, y] = 1 if np.random.rand() < 0.5 else 2

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

def update_oxygen(cell_grid, oxygen_grid, args):
    '''
    Updates the oxygen grid based on an oxygen diffusion equation. 
    '''
    normal_uptake = args.NORMAL_UPTAKE
    cancer_uptake = args.CANCER_UPTAKE
    diffusion = args.OXYGEN_DIFFUSION

    uptake = np.zeros_like(oxygen_grid)
    uptake[cell_grid == 1] = normal_uptake
    uptake[cell_grid == 2] = cancer_uptake
    uptake[cell_grid == 3] = 0

    updated_oxygen = np.clip(oxygen_grid + diffusion * laplace(oxygen_grid) - uptake, 0, 1)

    if np.random.rand() < 0.01:  # Reintroduce oxygen randomly
        updated_oxygen = np.clip(updated_oxygen + 0.001, 0, 1)  # Oxygen replenishment

    return updated_oxygen

def simulate_growth(args):
    '''
    Runs the tumor growth simulation based on stated cellular automata rules.
    '''
    N = args.GRID_SIZE
    T = args.TIME_STEPS

    cell_grid, oxygen_grid = initialize_grid(N, args)
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
                    #local_oxygen_threshold = args.CT1 if cancer_cell_neighbors > normal_cell_neighbors else args.CT2
                    if oxygen_grid[x, y] > args.CT1: 
                        new_grid[x, y] = 2
                        quiescent_clock[x, y] = 0
                    else:
                        quiescent_clock[x, y] += 1
                        if quiescent_clock[x, y] > args.QUIESCENCE_TIME:  
                            new_grid[x, y] = 0


        cell_grid = new_grid
        oxygen_grid = update_oxygen(cell_grid, oxygen_grid, args)

    final_cancer_cells = np.sum(cell_grid == 2) + np.sum(cell_grid == 3)
    return cell_grid, final_cancer_cells

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