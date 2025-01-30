import math
import numpy as np
import matplotlib.pyplot as plt
import tumor_growth as tg
from argparse import ArgumentParser, RawTextHelpFormatter
import pandas as pd
from avalanche_sizes import parse_args
from scipy.optimize import curve_fit
import os
import pickle

def func_log(x, a, b):
    """Logarithmic form of function for fitting"""
    return a + b * np.log(x)

def func_exp_log(x, a, b, c):
    """Fit function exp of polynomial plus log """
    return math.e**(a + b * 1/x + c * np.log(x))

def plot_tumor_sizes_vs_ratio(df, T, N):
    # Plotting the results

    # Divide p by 8 to adjust for number of neighbors
    df['ratio'] = df['ratio']/8
    df['p'] = df['p']/8

    #plt.plot(grouped_df.index, grouped_df['tumor_size'], linestyle='-', color='darkgreen', label='avg # tumor cells')
    #plt.plot(grouped_df.index, grouped_df['death_size'], linestyle='-', color='black', label='avg # death cells')
    plt.scatter(df['ratio'], df['tumor_size'], marker='o', color='green', alpha=0.7, label="simulated data")
    #plt.scatter(df['ratio'], df['death_size'], marker='x', color='black', alpha=0.7, label="dead")

    # Fit logarithmic curve
    popt, pcov = curve_fit(func_exp_log, df['ratio'], df['tumor_size'], maxfev=2000)
    print(f"fit: a = {popt[0]}, b = {popt[1]}, c={popt[1]}")

    # calculate new x's and y's
    x_fit = np.linspace(df['ratio'].min(), df['ratio'].max(), 50)
    y_fit = func_exp_log(x_fit, *popt)

    plt.plot(x_fit, y_fit, label=f"Fit \n"
                                 f"(exp({popt[0]:.2f} * {popt[1]:.2f}/x + {popt[2]:.2f} * log(x))")

    # calculate derivative
    dy_dx = np.gradient(y_fit) / np.gradient(x_fit) # TODO: check this! 
    plt.plot(x_fit, dy_dx, label=f"Derivative")

    plt.xscale('log')
    plt.axvline(x=1, color='red', linestyle='--', label='Theoretical Critical Point (p = d)')
    plt.xlabel('Growth-to-Death Probability Ratio (p/d)')
    plt.ylabel('Final Tumor Size (% of grid size)')
    plt.title("Number of Tumorous Cells vs. Growth-to-Death Probability Ratio \n"
              f"(T = {T}, # Experiments = 10)")
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.savefig(f"data/Tumorsize_ratio_T{T}_N{N}_log.png")

def simulate_tumor_size_different_ratios(params, N, T, num_it):
    """Simulate the tumor growth and store the final number of tumor and death cells in a dataframe."""
    # Run simulations for each growth/death ratio
    results = []
    for p, d in params:
        for i in range(num_it):
            print(f"p = {p}, d={d}, ratio={np.round(p / d, 3)}")
            grid = tg.simulate_growth(N, T, p, d, 0)
            tumor_size = np.sum(grid == 1)
            death_size = np.sum(grid == 2)
            results.append({
                "p": p,
                "d": d,
                "ratio": p / d,
                "tumor_size": tumor_size / (N * N),
                "death_size": death_size / (N * N)
            })

    return pd.DataFrame(results)

if __name__ == '__main__':
    # Parameters
    args = parse_args()
    # Parameters
    N = args.GRID_SIZE  # Size of the grid (default 50x50)
    T = args.TIME_STEPS  # Number of simulation steps

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
        growth_probabilities = args.GROWTH_PROBABILITIES  # values assumed for growth probability
        death_probabilities = args.DEATH_PROBABILITIES  # values assumed for death probability
        num_it = args.NUMBER_OF_EXPERIMENTS  # Number of times to repeat one experiment

        assert len(growth_probabilities) == len(death_probabilities), "Number of growth and death " \
                                                                      "probability values must be equal."
        assert num_it >= 1, "Must perform at least one experiment."

        # Results storage
        params = zip(growth_probabilities, death_probabilities)
        nparams = len(growth_probabilities)
        # Data not already generated, start simulation
        df = simulate_tumor_size_different_ratios(params, N, T, num_it)
        filepath = f"data/input/phase_transition_tumor_size_N{N}_T{T}_Np{nparams}_numit{num_it}.pkl"
        directory = os.path.dirname(filepath)
        if directory:  # Avoid creating a directory if input_file has no directory specified
            os.makedirs(directory, exist_ok=True)
        df.to_pickle(filepath)

    df.to_csv(f"data/input/phase_transition_tumor_size_N{N}_T{T}.csv", sep="\t")
    df['ratio'].to_csv(f"data/input/phase_transition_ratio_N{N}_T{T}.csv", index=False)
    df['tumor_size'].to_csv(f"data/input/tumor_size_N{N}_T{T}.csv", index=False)

    plot_tumor_sizes_vs_ratio(df, T, N)



