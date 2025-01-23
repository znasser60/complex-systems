import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser, RawTextHelpFormatter
import tumor_growth as tg
from scipy.optimize import curve_fit
from scipy.stats import kstest


def parse_args():
    "Parses inputs from commandline and returns them as a Namespace object."

    parser = ArgumentParser(prog = 'python3 avalanche_sizes.py',
        formatter_class = RawTextHelpFormatter, description =
        '  Plot the distribution of avalanche sizes (=number of cell state changes in one iteration) \n'
        '  and fit a powerlaw curve through the data. Evaluates the goodness of fit using Kolmogorov-Smirnov \n'
        '  test and p-value.\n\n'
        '  Example syntax:\n'
        '    python3 avalanche_sizes.py -N 50 -T 100 -p 0.2 -d 0.05\n')

    # Optionals
    parser.add_argument('-N', dest='GRID_SIZE', type=int, default=100,
        help='grid size (default = 100)')

    parser.add_argument('-T', dest='TIME_STEPS', type=int, default=500,
        help='number of time steps (default = 500)')

    parser.add_argument('-p', dest='GROWTH_PROBABILITY', type=float, default=0.3,
        help='growth probability, probability that tumor cell divides, (default = 0.3) ')

    parser.add_argument('-d', dest='DEATH_PROBABILITY', type=float, default=0.05,
                        help='death probability, probability that tumor cell dies, (default = 0.05) ')
    return parser.parse_args()

def simulate_growth_with_avalanches(N, T, p, d):
    """Simulate one timestep of the tumor growth at a time and count the number of death or growth events. Return
    all avalanche sizes in a list."""
    # Initialize the grid
    old_grid = np.zeros((N, N), dtype=int)
    center = N // 2
    old_grid[center, center] = 1  # Single tumor cell in the center
    avalanche_sizes = []
    # Tumor growth simulation
    for step in range(T):
        new_grid = old_grid.copy()
        new_grid = tg.update_grid(old_grid, new_grid, p, d)
        changes = np.sum(old_grid != new_grid)
        if changes > 0:
            avalanche_sizes.append(changes)
        if (step % 25 == 0):  # Save every 25th step
            tg.save_frame(old_grid, step)
            print(f"T = {step}, Number of changes: {changes}")
        old_grid = new_grid
        if np.all(old_grid == 0) or np.all(old_grid==1) or np.all(old_grid == 2):
            print(f"Simulation converged after {step} iterations!")
            break

    return avalanche_sizes, step

def func_powerlaw(x, a, b):
    """Powerlaw function: a*x**(-b)"""
    return a * x**(-b)

def plot_avalanche_distribution(avalanche_sizes, nit, N, T, p, d):
    """Plots the distribution of avalanche sizes and the fit of the powerlaw."""
    unique, counts = np.unique(avalanche_sizes, return_counts=True)
    popt, pcov = curve_fit(func_powerlaw, unique, counts, maxfev=2000)
    print(f"fit: a = {popt[0]}, b = {popt[1]}")

    # Perform KS test
    def cdf_powerlaw(x):
        """Theoretical CDF for the fitted power-law model."""
        theoretical_pdf = func_powerlaw(unique, *popt)
        theoretical_pdf /= np.sum(theoretical_pdf)  # Normalize to get probabilities
        theoretical_cdf = np.cumsum(theoretical_pdf)
        return np.interp(x, unique, theoretical_cdf)  # Interpolate CDF for all x values

    ks_statistic, p_value = kstest(avalanche_sizes, cdf_powerlaw)
    print(f"Kolmogorov-Smirnov statistic: {ks_statistic}")
    print(f"P-value: {p_value}")

    plt.figure(figsize=(8, 6))
    plt.loglog(unique, counts, marker='o', linestyle='none', markersize=5, alpha=0.7, label="Simulated Data")
    plt.loglog(unique, func_powerlaw(unique, *popt), linestyle='-', color='red', label=f'Power Law Fit ({popt[0]:.2f} x ^(-{popt[1]:.2f}))')

    plt.xlabel('Avalanche Size (Number of Cell Changes)')
    plt.ylabel('Frequency')
    plt.title('Power-Law Fit to Avalanche Size Distribution \n'
              f'(T={T}, N={N}, p ={p}, d={d}, #Experiments ={nit}) \n'
              f'KS Statistic: {ks_statistic:.4f}, P-value: {p_value:.2e}')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"data/avalanche_sizes_dist_N{N}_T{T}_p{p}_d{d}_nit_{nit}_loglog.png")
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    # Parameters
    N = args.GRID_SIZE  # Size of the grid (default 50x50)
    T = args.TIME_STEPS  # Number of simulation steps
    p = args.GROWTH_PROBABILITY  # Probability of tumor cell division
    d = args.DEATH_PROBABILITY  # Probability that a tumor cell dies
    all_sizes = []
    num_it = 20

    # Repeated experiments to generate data
    for i in range(num_it):
        avalanche_sizes, step = simulate_growth_with_avalanches(N, T, p, d)
        all_sizes += avalanche_sizes

    plot_avalanche_distribution(all_sizes, num_it, N, T, p, d)
    # TODO (optional): plot for several p and d


