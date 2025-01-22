import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from argparse import ArgumentParser, RawTextHelpFormatter
import os

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

    parser.add_argument('-T', dest='TIME_STEPS', type=int, default=100,
        help='number of time steps (default = 100)')

    parser.add_argument('-p', dest='GROWTH_PROBABILITY', type=float, default=0.3,
        help='growth probability, probability that tumor cell divides, (default = 0.3) ')

    parser.add_argument('-d', dest='DEATH_PROBABILITY', type=float, default=0.01,
                        help='death probability, probability that tumor cell dies, (default = 0.01) ')
    
    parser.add_argument('-m', dest = 'MUTATION_PROBABILITY', type = float, default = .00001, 
                        help = 'mutation probability of a healthy cell into a cancer cell, (default = .00001)')
    return parser.parse_args()


# Define neighborhood rule (Moore neighborhood)
def get_neighbors(x, y, N):
    neighbors = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if not (i == 0 and j == 0):
                nx, ny = x + i, y + j
                if 0 <= nx < N and 0 <= ny < N:
                    neighbors.append((nx, ny))
    np.random.shuffle(neighbors)
    return neighbors

# Probability of a healthy cell mutating into a cancer cell 
def mutate(p_mutate): 
    return 1 if np.random.rand() < p_mutate else 0


def save_frame(grid, step):
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

def simulate_growth(N, T, p, d, m, save_plots = False):
    # Initialize the grid
    old_grid = np.zeros((N, N), dtype=int)
    center = N // 2
    #old_grid[center, center] = 1  # Single tumor cell in the center
    # Tumor growth simulation
    for step in range(T):
        new_grid = old_grid.copy()
        for x, y in np.random.permutation([(i, j) for i in range(N) for j in range(N)]):
            if old_grid[x, y] == 1:  # If I am a tumor cell
                neighbors = get_neighbors(x, y, N)
                for nx, ny in neighbors:
                    # If neighbors are healthy
                    if old_grid[nx, ny] == 0 and np.random.rand() < p:
                        new_grid[nx, ny] = 1 #Infect each neighbor with growth probability
            if old_grid[x, y] == 1 and np.random.rand() < d:
                # this tumor cell dies with d probability
                new_grid[x, y] = 2
            elif old_grid[x,y] == 0 : 
                new_grid[x,y] = mutate(m)
        old_grid = new_grid
        #print(f"Number of tumor cells (NEW GRID): {np.sum(new_grid == 1)}, healthy cells: {np.sum(new_grid == 0)}, "
        #      f"dead cells: {np.sum(new_grid == 2)}, total cells: {N * N}")
        if (step % 5 == 0) and save_plots:  # Save every 5th step
            save_frame(old_grid, step)
    return old_grid

def save_gif(T):
    # Create GIF from saved frames
    output_dir = os.path.join(os.getcwd(), 'data')
    images = [Image.open(os.path.join(output_dir, f'frame_{step}.png')) for step in range(0, T, 5)]
    gif_path = os.path.join(output_dir, 'tumor_growth_simulation.gif')
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
    grid = simulate_growth(N, T, p, d, m, save_plots = True)
    save_gif(T)

