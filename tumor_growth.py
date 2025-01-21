import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Parameters
GRID_SIZE = 50  # Size of the grid (50x50)
TIME_STEPS = 100  # Number of simulation steps
GROWTH_PROBABILITY = 0.3  # Probability of tumor cell division

# Initialize the grid
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
center = GRID_SIZE // 2
grid[center, center] = 1  # Single tumor cell in the center

# Define neighborhood rule (Moore neighborhood)
def get_neighbors(x, y, grid):
    neighbors = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if not (i == 0 and j == 0):
                nx, ny = x + i, y + j
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    neighbors.append((nx, ny))
    return neighbors

# Store frames for the GIF
frames = []

def save_frame(grid, step):
    plt.figure(figsize=(5, 5))
    plt.imshow(grid, cmap='viridis', interpolation='nearest')
    plt.axis('off')
    plt.title(f"Tumor Growth at Step {step}")
    # Save the current frame to an image object
    plt.savefig(f'C:/Users/gelie/Home/ComputationalScience/ComplexSystems/project/complex-systems/data/frame_{step}.png', bbox_inches='tight', pad_inches=0)
    plt.close()

# Tumor growth simulation
for step in range(TIME_STEPS):
    new_grid = grid.copy()
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if grid[x, y] == 1:  # Tumor cell
                neighbors = get_neighbors(x, y, grid)
                for nx, ny in neighbors:
                    if grid[nx, ny] == 0 and np.random.rand() < GROWTH_PROBABILITY:
                        new_grid[nx, ny] = 1
            elif grid[x, y] == 1 and np.random.rand() < 0.05:
                new_grid[x, y] = 2
    grid = new_grid
    if step % 5 == 0:  # Save every 5th step
        save_frame(grid, step)

# Create GIF from saved frames
images = [Image.open(f'C:/Users/gelie/Home/ComputationalScience/ComplexSystems/project/complex-systems/data/frame_{step}.png') for step in range(0, TIME_STEPS, 5)]
gif_path = 'C:/Users/gelie/Home/ComputationalScience/ComplexSystems/project/complex-systems/data/tumor_growth_simulation.gif'
images[0].save(gif_path, save_all=True, append_images=images[1:], duration=300, loop=0)

# Display the path to download the GIF
print(f"GIF saved as: {gif_path}")
