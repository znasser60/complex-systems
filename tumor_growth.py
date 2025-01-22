import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    parser.add_argument('-T', dest='TIME_STEPS', type=int, default=100, help='Number of time steps (default: 100)')
    parser.add_argument('-p', dest='GROWTH_PROBABILITY', type=float, default=0.6, help='Tumor cell division probability (default: 0.6)')
    parser.add_argument('-od', dest='OXYGEN_DIFFUSION', type=float, default=0.3, help='Oxygen diffusion coefficient (default: 0.3)')
    parser.add_argument('-cu', dest='CANCER_UPTAKE', type=float, default=0.1, help='Cancer cell oxygen uptake rate (default: 0.1)')
    parser.add_argument('-nu', dest='NORMAL_UPTAKE', type=float, default=0.2, help='Normal cell oxygen uptake rate (default: 0.2)')
    parser.add_argument('-o', dest='INITIAL_OXYGEN', type=float, default=0.8, help='Initial oxygen concentration (default: 0.8)')
    parser.add_argument('-nt1', dest='NT1', type=float, default=0.05, help='Oxygen threshold for normal cells with normal neighbors (default: 0.05)')
    parser.add_argument('-nt2', dest='NT2', type=float, default=0.1, help='Oxygen threshold for normal cells with cancer neighbors (default: 0.1)')
    parser.add_argument('-ct1', dest='CT1', type=float, default=0.05, help='Oxygen threshold for cancer cells with cancer neighbors (default: 0.05)')
    parser.add_argument('-ct2', dest='CT2', type=float, default=0.1, help='Oxygen threshold for cancer cells with normal neighbors (default: 0.1)')
    parser.add_argument('-qt', dest='QUIESCENCE_TIME', type=int, default=10, help='Quiescence time for cancer cells (default: 10)')
    parser.add_argument('--output-path', dest='OUTPUT_PATH', type=str, default='./', help='Path to save frames and GIF')
    return parser.parse_args()

def get_neighbors(x, y, N):
    neighbors = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if not (i == 0 and j == 0):
                nx, ny = x + i, y + j
                if 0 <= nx < N and 0 <= ny < N:
                    neighbors.append((nx, ny))
    return neighbors

def count_neighbors(x, y, cell_grid, N):
    """Counts the number of normal and cancer neighbors of a cell."""
    neighbors = get_neighbors(x, y, N)
    normal = sum(1 for nx, ny in neighbors if cell_grid[nx, ny] == 1)
    cancer = sum(1 for nx, ny in neighbors if cell_grid[nx, ny] == 2)
    return normal, cancer

def update_oxygen(cell_grid, oxygen_grid, normal_uptake, cancer_uptake, diffusion):
    """Updates the oxygen grid."""
    uptake = np.zeros_like(oxygen_grid)
    uptake[cell_grid == 1] = normal_uptake
    uptake[cell_grid == 2] = cancer_uptake
    return np.clip(oxygen_grid + diffusion * laplace(oxygen_grid) - uptake, 0, 1)

def save_frame(cell_grid, oxygen_grid, step, output_path):
    """Saves a frame of the simulation."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cell_grid, cmap='viridis', interpolation='nearest')
    plt.title(f"Cell States at Step {step}")
    plt.axis('off')
    plt.legend(handles=[
        mpatches.Patch(color='black', label='Quiescent Cancer Cell'),
        mpatches.Patch(color='yellow', label='Cancer Cell'),
        mpatches.Patch(color='green', label='Normal Cell'),
        mpatches.Patch(color='purple', label='Empty')
    ], loc='upper right', fontsize=8)

    plt.subplot(1, 2, 2)
    plt.imshow(oxygen_grid, cmap='coolwarm', interpolation='nearest')
    plt.title(f"Oxygen Levels at Step {step}")
    plt.axis('off')
    cbar = plt.colorbar()
    cbar.set_label('Oxygen Concentration', fontsize=8)
    plt.savefig(f"{output_path}/frame_{step}.png", bbox_inches='tight', pad_inches=0)
    plt.close()

def simulate_growth(args):
    """Runs the tumor growth simulation."""
    N = args.GRID_SIZE
    T = args.TIME_STEPS

    # Initialize grids
    cell_grid = np.zeros((N, N), dtype=int)
    oxygen_grid = np.full((N, N), args.INITIAL_OXYGEN)
    quiescent_clock = np.zeros((N, N), dtype=int)

    # Add initial cells and vessels
    center = N // 2
    cell_grid[center, center] = 2  # Initial cancer cell
    cell_grid[40,10] = 2  
    cell_grid[41,11] = 1 
    vessel_positions = [(10, 10), (25, 25), (40, 40)] # Random vessel placement (TO BE CHANGED)
    for x, y in vessel_positions: 
        oxygen_grid[x, y] = 1.0

    for step in range(T):
        new_grid = cell_grid.copy()
        for x in range(N):
            for y in range(N):
                if cell_grid[x, y] == 1: # Normal cell
                    normal_cell_neighbors, cancer_cell_neighbors = count_neighbors(x,y,cell_grid,N)
                    if normal_cell_neighbors > cancer_cell_neighbors:
                        local_oxygen_threshold = args.NT1
                    else: 
                        local_oxygen_threshold = args.NT2

                    if oxygen_grid[x, y] < local_oxygen_threshold:  # Normal cell dies if oxygen is under local threshold
                        new_grid[x, y] = 0
                    else:  # Attempts to divide
                        neighbors = get_neighbors(x, y, N)
                        empty_neighbors = [(nx, ny) for nx, ny in neighbors if cell_grid[nx, ny] == 0]
                        if empty_neighbors and np.random.rand() < args.GROWTH_PROBABILITY:
                            nx, ny = empty_neighbors[np.random.choice(len(empty_neighbors))]
                            new_grid[nx, ny] = 1 

                elif cell_grid[x, y] == 2:  # Cancer cell
                    normal_cell_neighbors, cancer_cell_neighbors = count_neighbors(x,y,cell_grid,N)
                    if cancer_cell_neighbors > normal_cell_neighbors: 
                        local_oxygen_threshold = args.CT1
                    else: 
                        local_oxygen_threshold = args.CT2

                    if oxygen_grid[x, y] < local_oxygen_threshold:  # Cancer cell becomes quiescent if oxgen is under set threshold
                        new_grid[x, y] = 3 
                    else:  # Attempts to divide
                        neighbors = get_neighbors(x, y, N)
                        empty_neighbors = [(nx, ny) for nx, ny in neighbors if cell_grid[nx, ny] == 0]
                        if empty_neighbors and np.random.rand() < args.GROWTH_PROBABILITY:
                            nx, ny = empty_neighbors[np.random.choice(len(empty_neighbors))]
                            new_grid[nx, ny] = 2

                elif cell_grid[x, y] == 3:  # Quiescent cancer cell
                    normal_cell_neighbors, cancer_cell_neighbors = count_neighbors(x,y,cell_grid,N)
                    if cancer_cell_neighbors > normal_cell_neighbors: 
                        local_oxygen_threshold = args.CT1
                    else: 
                        local_oxygen_threshold = args.CT2
                    
                    if oxygen_grid[x, y] > local_oxygen_threshold:  # Revives if oxygen increases above threshold
                        new_grid[x, y] = 2
                        quiescent_clock[x, y] = 0
                    else: 
                        quiescent_clock[x, y] += 1
                        if quiescent_clock[x, y] > args.QUIESCENCE_TIME:  # Cancer cell dies after too much time without enough oxygen 
                            new_grid[x, y] = 0
        cell_grid = new_grid
        oxygen_grid = update_oxygen(cell_grid, oxygen_grid, args.NORMAL_UPTAKE, args.CANCER_UPTAKE, args.OXYGEN_DIFFUSION)
        
        if step % 5 == 0:  # Save every 5th step
            save_frame(cell_grid, oxygen_grid, step, args.OUTPUT_PATH)

    return cell_grid

def save_gif(output_path, time_steps):
    """Creates a GIF from the saved frames."""
    images = [Image.open(f'{output_path}/frame_{step}.png') for step in range(0, time_steps, 5)]
    gif_path = f'{output_path}/tumor_growth_simulation.gif'
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=300, loop=0)

    print(f"GIF saved as: {gif_path}")

if __name__ == '__main__':
    args = parse_args()
    simulate_growth(args)
    save_gif(args.OUTPUT_PATH, args.TIME_STEPS)
