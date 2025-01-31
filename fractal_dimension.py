import numpy as np
import matplotlib.pyplot as plt
from tumor_growth import simulate_growth, save_gif
from avalanche_sizes import parse_args


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

    return -coeffs  # The slope gives the fractal dimension


if __name__ == '__main__':
    args = parse_args()
    # Parameters
    N = args.GRID_SIZE  # Size of the grid
    T = args.TIME_STEPS  # Number of simulation steps
    num_it = args.NUMBER_OF_EXPERIMENTS # Number of experiments per parameter configuration
    death_probabilities = args.DEATH_PROBABILITIES
    growth_probabilities = args.GROWTH_PROBABILITIES

    assert len(death_probabilities) == len(growth_probabilities), f"Ngrowth: {len(growth_probabilities)} " \
                                                                  f"vs. Ndeath: {len(death_probabilities)}"

    pd_params = zip(growth_probabilities, death_probabilities)

    # Plot fractal dimension vs growth/death ratio
    plt.figure(figsize=(8, 6))

    for i in range(num_it):
        fractal_dimensions = []

        for i in range(len(growth_probabilities)):
            p = growth_probabilities[i]
            d = death_probabilities[i]
            print(f"p = {p}, d={d}, ratio={np.round(p / d, 3)}")
            # Repeated experiments to generate data
            grid = simulate_growth(N, T, p, d, 0, save_plots=False)
            grid[grid == 1] = 2 # Set tumor cells equal to death cells

            # Compute fractal dimension
            fd = fractal_dimension(grid)
            fractal_dimensions.append(fd)
        print(f"fractal_dimensions: {len(fractal_dimensions)}, growth prob: {len(growth_probabilities)}")
        plt.plot(np.array(growth_probabilities) / np.array(death_probabilities), fractal_dimensions,
             marker='o', color="blue", alpha=0.6)
    plt.xlabel("Growth/Death Probability Ratio (p/d)")
    plt.xscale('log')
    plt.ylabel("Fractal Dimension")
    plt.title("Fractal Dimension vs. Growth/Death Probability Ratio")
    plt.grid(True)
    plt.savefig(f"data/fractal_dimension_T{T}_N{N}.png")


