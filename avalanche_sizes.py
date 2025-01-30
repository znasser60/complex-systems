import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser, RawTextHelpFormatter
import tumor_growth as tg
from scipy.optimize import curve_fit
from scipy.stats import kstest
import pickle
import pandas as pd
import os


def parse_args():
    "Parses inputs from commandline and returns them as a Namespace object."

    parser = ArgumentParser(prog = 'python3 avalanche_sizes.py',
        formatter_class = RawTextHelpFormatter, description =
        '  Plot the distribution of avalanche sizes (=number of cell state changes in one iteration) \n'
        '  and fit a powerlaw curve through the data. Evaluates the goodness of fit using Kolmogorov-Smirnov \n'
        '  test and p-value.\n\n'
        '  Example syntax:\n'
        '    python3 avalanche_sizes.py -N 50 -T 100 -p 0.2 0.2 0.5 0.5 -d 0.05 0.1 0.05 0.1 -num_it 2\n'
        '    python3 avalanche_sizes.py -input data/input/mydata.pkl -N 50 -T 100')

    # Optionals
    parser.add_argument('-input', dest='INPUT_FILEPATH', help='(Optional) Path to the input file (pickle).')
    parser.add_argument('-N', dest='GRID_SIZE', type=int, default=100,
        help='grid size (default = 100)')

    parser.add_argument('-T', dest='TIME_STEPS', type=int, default=500,
        help='number of time steps (default = 500)')

    parser.add_argument('-p', dest='GROWTH_PROBABILITIES', type=float, nargs='+',
        help='number of values assumed for growth probability, probability that tumor cell divides')

    parser.add_argument('-d', dest='DEATH_PROBABILITIES', type=float, nargs='+',
                        help='Values assumed for death probability, probability that tumor cell dies ')

    parser.add_argument('-num_it', dest="NUMBER_OF_EXPERIMENTS", type=int, default=2,
                        help="Number of times an experiment with the same set of parameters is repeated (default = 2).")

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
        new_grid = tg.update_grid(old_grid, new_grid, p, d, 0)
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

def perform_kstest(avalanche_sizes, popt):
    # Perform KS test
    def cdf_powerlaw(x):
        """Theoretical CDF for the fitted power-law model."""
        theoretical_pdf = func_powerlaw(x, *popt)
        theoretical_pdf /= np.sum(theoretical_pdf)  # Normalize to get probabilities
        theoretical_cdf = np.cumsum(theoretical_pdf)
        return theoretical_cdf

    ks_statistic, p_value = kstest(avalanche_sizes, cdf_powerlaw)
    print(f"Kolmogorov-Smirnov statistic: {ks_statistic}")
    print(f"P-value: {p_value}")
    return ks_statistic, p_value


def plot_avalanche_distribution(avalanche_sizes, nit, N, T, p, d):
    """Plots the distribution of avalanche sizes and the fit of the powerlaw."""
    unique, counts = np.unique(avalanche_sizes, return_counts=True)
    unique = unique * 8
    popt, pcov = curve_fit(func_powerlaw, unique, counts, maxfev=2000)
    print(f"fit: a = {popt[0]}, b = {popt[1]}")

    ks_statistic, p_value = perform_kstest(avalanche_sizes, popt)

    plt.figure(figsize=(8, 6))
    plt.loglog(unique, counts, marker='o', linestyle='none', markersize=5, alpha=0.7, label="Simulated Data")
    plt.loglog(unique, func_powerlaw(unique, *popt), linestyle='-', color='red', label=f'Power Law Fit ({popt[0]:.2f} x ^(-{popt[1]:.2f}))')

    plt.xlabel('Avalanche Size (Number of Cell Changes)')
    plt.ylabel('Frequency')
    plt.title('Power-Law Fit to Avalanche Size Distribution \n'
              f'(T={T}, N={N}, p ={p*8}, d={d}, #Experiments ={nit}) \n'
              f'KS Statistic: {ks_statistic:.4f}, P-value: {p_value:.2e}')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"data/avalanche_sizes_dist_N{N}_T{T}_p{p}_d{d}_nit_{nit}_loglog.png")

def simulate_multiple_p(N, T, growth_probabilities, death_probabilities, num_it):
    """Take Np different parameters of growth probability and get the distribution of the avalanche sizes.
    Fit a power law through the distribution and plot the curve and the goodness-of-fit statistics in one plot."""

    pd_params = zip(growth_probabilities, death_probabilities)

    all_sizes = []
    results = []

    for p, d in pd_params:
        print(f"p = {p}, d={d}, ratio={np.round(p/d, 3)}")
        # Repeated experiments to generate data
        for i in range(num_it):
            avalanche_sizes, step = simulate_growth_with_avalanches(N, T, p, d)
            all_sizes += avalanche_sizes
        unique, counts = np.unique(all_sizes, return_counts=True)
        max_avalanches = max(counts) if len(counts) > 0 else 0
        try:
            popt, pcov = curve_fit(func_powerlaw, unique, counts, maxfev=2000)
            print(f"p = {p}, d={d}, fit: a = {popt[0]}, b = {popt[1]}")
            ks_statistic, p_value = perform_kstest(all_sizes, popt)
            results.append({
                "p": p,
                "d": d,
                "ratio": p/d,
                "ks_statistic": ks_statistic,
                "p_value": p_value,
                "a": popt[0],
                "b": popt[1],
                "max_avalanches": max_avalanches
            })
        except Exception:
            # Need at least two data points to fit model
            if len(unique) < 2 :
                print("Error: Not enough data to perform model fit!")
                continue
            else :
                print("An Error occured.")
                break

    return pd.DataFrame(results)


def plot_fits_multiple_p(df):
    """Plot multiple power law fits for multiple ratios of growth and death probability."""
    min_avalanches = 1
    max_avalanches = df['max_avalanches'].max()
    x = range(min_avalanches, max_avalanches)
    plt.figure(figsize=(8, 6))
    for _, row in df.iterrows():
        a = row['a']
        b = row['b']
        D = row['ks_statistic']
        p_value = row['p_value']
        ratio = row['ratio'] #/8 rescale according to number of neighbors
        d = row['d']
        p = row['p'] #/8 rescale according to number of neighbors
        plt.loglog(x, func_powerlaw(x, a, b), linestyle='-',
                   label=f'd = {d}, ratio={np.round(ratio, 2)}, \n'
                         f'D= {D:.2f}, p-value = {p_value:.2e}, ({a:.2f} x ^(-{b:.2f}))')
    plt.xlabel('Avalanche Size (Number of Cell Changes)')
    plt.ylabel('Frequency')
    plt.title('Power-Law Fit of Avalanche Sizes for Different Growth-to-Death Ratios \n'
              f'(T={T}, N={N}, p={p}, #Experiments ={num_it}) \n')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"data/avalanche_sizes_multiple_d_constant_p_N{N}_T{T}_nit_{num_it}_loglog.png")


def analyze_avlanche_distribution(N, T, p, d, num_it):
    """Plot the avalanche size distribution for single config of p and d parameters. Perform num_it experiments.
    Plot the data in a scatter plot. Fit a powerlaw curve on top of the data and plot the curve and goodness-of-fit
    statistics."""

    all_sizes = []
    print(f"p: {p}, d: {d}")
    # Repeated experiments to generate data
    for i in range(num_it):
        avalanche_sizes, step = simulate_growth_with_avalanches(N, T, p, d)
        all_sizes += avalanche_sizes
    plot_avalanche_distribution(all_sizes, num_it, N, T, p, d)


if __name__ == '__main__':
    args = parse_args()
    # Parameters
    N = args.GRID_SIZE  # Size of the grid (default 50x50)
    T = args.TIME_STEPS  # Number of simulation steps
    growth_probabilities = args.GROWTH_PROBABILITIES  # Number of values assumed for growth probability
    death_probabilities = args.DEATH_PROBABILITIES  # Number of values assumed for death probability
    num_it = args.NUMBER_OF_EXPERIMENTS # Number of times to repeat one experiment

    assert num_it >= 1, "Must perform at least one experiment."

    if growth_probabilities is not None and death_probabilities is not None \
            and len(growth_probabilities) == len(death_probabilities) == 1 :
        analyze_avlanche_distribution(N, T, growth_probabilities[0], death_probabilities[0], num_it)
    else :
        if args.INPUT_FILEPATH is not None:
            input_filepath = args.INPUT_FILEPATH
            # Check if the pickle file exists
            if os.path.exists(input_filepath):
                # Load data from pickle
                with open(input_filepath, "rb") as f:
                    df = pickle.load(f)
                print("Data loaded from pickle.")
            else:
                print("Error: Input File does not exist!")
        else:
            assert len(growth_probabilities) == len(death_probabilities), "Number of growth and death " \
                                                                          "probability values must be equal."

            # Data not already generated, start simulation
            df = simulate_multiple_p(N, T, growth_probabilities, death_probabilities, num_it)
            filepath = "data/input/avalanche_sizes_constant_p_different_d.pkl"
            directory = os.path.dirname(filepath)
            if directory:  # Avoid creating a directory if input_file has no directory specified
                os.makedirs(directory, exist_ok=True)
            df.to_pickle(filepath)

        plot_fits_multiple_p(df)




