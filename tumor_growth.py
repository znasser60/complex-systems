import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from argparse import ArgumentParser, RawTextHelpFormatter
import os

def parse_args():
    "Parses inputs from commandline and returns them as a Namespace object."

    parser = ArgumentParser(prog = 'python3 tumor_growth.py',
        formatter_class = RawTextHelpFormatter, description =
        '  Simulate tumor growth given arguments for growth, death and mutation probability, grid size and number of time steps.\n\n'
        '  Example syntax:\n'
        '    python3 tumor_growth.py -N 50 -T 100 -p 0.1 -d 0.01 -m 0 \n')

    # Optionals
    parser.add_argument('-N', dest='GRID_SIZE', type=int, default=50,
        help='grid size (default = 50)')

    parser.add_argument('-T', dest='TIME_STEPS', type=int, default=100,
        help='number of time steps (default = 100)')

    parser.add_argument('-p', dest='GROWTH_PROBABILITY', type=float, default=0.3,
        help='growth probability, probability that tumor cell divides, (default = 0.3) ')

    parser.add_argument('-d', dest='DEATH_PROBABILITY', type=float, default=0.1,
                        help='death probability, probability that tumor cell dies, (default = 0.1) ')
    
    parser.add_argument('-m', dest = 'MUTATION_PROBABILITY', type = float, default = .00001, 
                        help = 'mutation probability of a healthy cell into a cancer cell, (default = .00001)')
    return parser.parse_args()


# Define neighborhood rule (Moore neighborhood)
def get_neighbors(x, y, N):
    """Given coordinate x and y in a N by N grid, return the coordinates of all neighbors (Moore neighborhood)."""
    neighbors = []

    assert (0 <= x < N) and (0 <= y < N)

    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if not (i == 0 and j == 0):
                nx, ny = x + i, y + j
                if 0 <= nx < N and 0 <= ny < N:
                    neighbors.append((nx, ny))

    # Minimum 3 neighbors if we are in edge of grid
    assert 3 <= len(neighbors) <= 8
    return neighbors

# Probability of a healthy cell mutating into a cancer cell 
def mutate(m):
    """Return 1 if this cell mutates according to probability m. """
    return 1 if np.random.rand() < m else 0


def save_frame(grid, step):
    """Save a heatmap of the current grid state. """
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

def update_grid(old_grid, new_grid, p, d, m):
    """Iterate through the whole old grid and update new grid where a healthy cell becomes a tumor or
    a tumor cell from the old grid dies."""

    assert np.size(new_grid,1) == np.size(new_grid,0) == np.size(old_grid,1) == np.size(old_grid,0)

    N = np.size(old_grid,1)
    # Iterate over cells in random order
    for x, y in np.random.permutation([(i, j) for i in range(N) for j in range(N)]):
        if old_grid[x, y] == 1:  # If I am a tumor cell
           neighbors = get_neighbors(x, y, N)
           for nx, ny in neighbors:
               # If neighbors are healthy
               if old_grid[nx, ny] == 0 and np.random.rand() < p:
                   new_grid[nx, ny] = 1  # Infect each neighbor with growth probability
        elif old_grid[x, y] == 0: # If I am healthy, maybe mutate
           new_grid[x, y] = mutate(m)
        if (old_grid[x,y] == 1) and (np.random.rand() < d):  # If I am tumor, maybe die
            new_grid[x, y] = 2
    return new_grid

def simulate_growth(N, T, p, d, m, save_plots = False):
    """Perform T simulation steps. Returns the grid with the new cell values after T grid updates. If save_plots==True,
    save each 5th frame in a png file, else no images are created."""

    # Initialize the grid
    old_grid = np.zeros((N, N), dtype=int)
    center = N // 2
    old_grid[center, center] = 1  # Single tumor cell in the center

    # Tumor growth simulation
    for step in range(T):
        new_grid = old_grid.copy()
        old_grid = update_grid(old_grid, new_grid, p, d, m)
        if save_plots and step % 5 == 0:  # Save every 5th step
            save_frame(old_grid, step)
    return old_grid

def save_gif(T, N, p, d, m):
    """Create a gif, assuming that images have been created for each 5th time step of the simulation. """
    # Create GIF from saved frames
    output_dir = os.path.join(os.getcwd(), 'data')
    images = [Image.open(os.path.join(output_dir, f'frame_{step}.png')) for step in range(0, T, 5)]
    gif_path = os.path.join(output_dir, f'tumor_growth_simulation_T{T}_N{N}_p{p}_d{d}_m{m}.gif')
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=300, loop=0)

    # Display the path to download the GIF
    print(f"GIF saved as: {gif_path}")

if __name__ == '__main__':
    args = parse_args()
    # Parameters
    N = args.GRID_SIZE  # Size of the grid (default 50x50)
    T = args.TIME_STEPS  # Number of simulation steps
    p = args.GROWTH_PROBABILITY  # Probability of tumor cell division
    d = args.DEATH_PROBABILITY # Probability that a tumor cell dies
    m = args.MUTATION_PROBABILITY # probabilty that a healthy cell turns into a tumor cell

    assert 0 < p < 1, "Growth probability must be greater 0, less than 1"
    assert 0 < d < 1, "Death probability must be greater 0, less than 1"
    assert 0 <= m < 1, "Mutation probability must be greater or equal 0, less than 1"

    grid = simulate_growth(N, T, p, d, m, save_plots = True)
    save_gif(T, N, p, d, m)