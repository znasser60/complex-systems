import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from argparse import ArgumentParser, RawTextHelpFormatter
import os
from tumor_growth import simulate_growth, save_gif
from avalanche_sizes import parse_args
from scipy.optimize import curve_fit


def box_count(grid):
    """
    Compute the fractal dimension using the box-counting method.
    """
    sizes = np.unique(np.logspace(0.5, np.log2(min(grid.shape)), base=2, num=10).astype(int))

    counts = []

    for size in sizes:
        reshaped = grid[:grid.shape[0] // size * size, :grid.shape[1] // size * size]
        reshaped = reshaped.reshape((reshaped.shape[0] // size, size, reshaped.shape[1] // size, size))
        non_empty_boxes = np.sum(np.any(reshaped, axis=(1, 3)))
        counts.append(non_empty_boxes)

    return sizes, counts


def fractal_dimension(grid):
    """
    Compute the fractal dimension by fitting a power law to the box-counting data.
    """
    sizes, counts = box_count(grid)
    log_sizes = np.log(1 / sizes)
    log_counts = np.log(counts)
    coeffs, _ = np.polyfit(log_sizes, log_counts, 1)

    print(f"coeffs: {coeffs}")
    return -coeffs  # The slope gives the fractal dimension


if __name__ == '__main__':
    args = parse_args()
    # Parameters
    N = 50  # Size of the grid (default 50x50)
    T = 50  # Number of simulation steps
    num_it = 5
    ratios = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.16, 0.166666, 0.17, 0.18, 0.19, 0.2, 0.3, 0.5, 0.75, 1] #[0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9, 0.95, 1.0, 1.25, 1.5, 1.75, 2.0, 5.0, 10.0]
    ratios_rescaled = [ratio * 8 for ratio in ratios]
    death_probabilities = [0.1 for r in range(len(ratios_rescaled))] #[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.01, 0.01]
    growth_probabilities_rescaled = [ratios_rescaled[i] * death_probabilities[i] for i in range(len(ratios_rescaled))]
    print(f"p: {growth_probabilities_rescaled}, d: {death_probabilities}")
    pd_params = zip(growth_probabilities_rescaled, death_probabilities)

    # Plot fractal dimension vs growth/death ratio
    plt.figure(figsize=(8, 6))

    for i in range(num_it):
        fractal_dimensions = []

        for i in range(len(growth_probabilities_rescaled)):
            p = growth_probabilities_rescaled[i]
            d = death_probabilities[i]
            print(f"p = {p}, d={d}, ratio={np.round(p / d, 3)}")
            # Repeated experiments to generate data
            grid = simulate_growth(N, T, p, d, 0, save_plots=False)
            grid[grid == 1] = 2 # Set tumor cells equal to death cells

            # Compute fractal dimension
            fd = fractal_dimension(grid)
            fractal_dimensions.append(fd)
        print(f"fractal_dimensions: {len(fractal_dimensions)}, growth prob: {len(growth_probabilities_rescaled)}")
        plt.plot(np.array(growth_probabilities_rescaled) / np.array(death_probabilities), fractal_dimensions,
             marker='o', alpha=0.6)
    plt.xlabel("Growth/Death Probability Ratio (p/d)")
    plt.xscale('log')
    plt.ylabel("Fractal Dimension")
    plt.title("Fractal Dimension vs. Growth/Death Probability Ratio")
    plt.grid(True)
    plt.savefig("data/fractal_dimension.png")
    plt.show()
    plt.close()

