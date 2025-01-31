import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tumor_growth_with_oxygen import parse_args, simulate_growth

def box_counting_fractal_dimension(grid, min_box_size=1, max_box_size=50):
    """
    Compute the fractal dimension of the tumor grid using box-counting method.
    """
    sizes = []
    counts = []

    N = grid.shape[0]
    box_sizes = np.unique(np.logspace(np.log10(min_box_size), np.log10(max_box_size), num=10, dtype=int))

    for box_size in box_sizes:
        num_boxes = 0
        for x in range(0, N, box_size):
            for y in range(0, N, box_size):
                if np.any(grid[x:x+box_size, y:y+box_size]): 
                    num_boxes += 1

        sizes.append(1/box_size)
        counts.append(num_boxes)

    sizes = np.array(sizes)
    counts = np.array(counts)
    coeffs, _ = curve_fit(lambda x, a, b: a * x + b, np.log(sizes), np.log(counts))
    fractal_dimension = coeffs[0]
    return fractal_dimension

def calculate_fractal_dimensions(args, num_simulations=10):
    fractal_dimensions = []
    oxygen_levels = np.linspace(1e-6, 1e-3, 20)
    all_fractal_dimensions = []
    for _ in range(num_simulations):
        fractal_dimensions = []
        for oxygen in oxygen_levels: 
            args.INITIAL_OXYGEN = oxygen
            cell_grid, _ = simulate_growth(args, max_time=50)
            fractal_dim = box_counting_fractal_dimension(cell_grid)
            fractal_dimensions.append(fractal_dim)
                  
        all_fractal_dimensions.append(fractal_dimensions)
        plt.plot(oxygen_levels, fractal_dimensions, color='red', alpha=0.3)  

    avg_fractal_dimensions = np.mean(all_fractal_dimensions, axis=0)
    plt.plot(oxygen_levels, avg_fractal_dimensions, color='red', linewidth=2, label="Average")

    plt.xlabel("Oxygen Levels")
    plt.ylabel("Fractal Dimension")
    plt.title("Fractal Dimension vs Oxygen Levels at T=50")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    calculate_fractal_dimensions(args)