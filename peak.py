import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from argparse import ArgumentParser, RawTextHelpFormatter
import os
from itertools import product 
import pickle
from scipy.optimize import curve_fit

def parse_args_peaks():
    "Parses inputs from commandline and returns them as a Namespace object."

    parser = ArgumentParser(prog = 'python3 tumor_growth.py',
        formatter_class = RawTextHelpFormatter, description =
        '  Simulate tumor growth given arguments for growth probability, grid size and number of time steps.\n\n'
        '  Example syntax:\n'
        '    python3 tumor_growth.py -N 50 -T 100 -p 0.1\n')

    # Optionals
    parser.add_argument('-N', dest='GRID_SIZE', type=list, default= [ 300, 350, 400, 450, 500, 550, 600, 650 ]  ,
        help='grid size (default = 50)')

    parser.add_argument('-T', dest='TIME_STEPS', type=int, default=100,
        help='number of time steps (default = 100)')

    parser.add_argument('-p', dest='GROWTH_PROBABILITY', type=float, default= [.5/8] , 
        help='growth probability, probability that tumor cell divides, (default = 0.3) ')

    parser.add_argument('-d', dest='DEATH_PROBABILITY', type=list, default=[5e-05, 0.0001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.75, 1], 
                        help='death probability, probability that tumor cell dies, (default = 0.01) ')
    
    parser.add_argument('-m', dest = 'MUTATION_PROBABILITY', type = list, default = 0, 
                        help = 'mutation probability of a healthy cell into a cancer cell, (default = .00001)')
    return parser.parse_args()


# Define neighborhood rule (Moore neighborhood)
def get_neighbors(x, y, N):
    """
    Get the neighboring coordinates of a given cell in an N x N grid.

    Parameters:
    x (int): The x-coordinate of the cell.
    y (int): The y-coordinate of the cell.
    N (int): The size of the grid (N x N).

    Returns:
    list of tuple: A list of tuples representing the coordinates of the neighboring cells.
                   The neighbors are shuffled randomly.
    """
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
    """
    Determines whether a mutation occurs based on a given probability.
    Parameters:
    p_mutate (float): The probability of mutation, a value between 0 and 1.
    Returns:
    int: Returns 1 if a mutation occurs, otherwise returns 0.
    """

    return 1 if np.random.rand() < p_mutate else 0


def simulate_growth_peaks(N, T, p, d, m, k=5):
    """
    Runs the simulation k times for each (grid size, growth/death ratio) combination
    and stores the average and standard deviation of the final tumor size.

    Parameters:
    - N (list): List of grid sizes
    - T (int): Number of time steps
    - p (list): List of growth probabilities
    - d (list): List of death probabilities
    - m (float): Mutation probability
    - k (int): Number of repetitions per parameter set

    Returns:
    - results (dict): {grid_size: {ratio: (mean_final_tumor_size, std_final_tumor_size)}}
    """
    results = {}

    for size in N:
        print('GRID SIZE ' , size)
        
        neighbors_map = {
        (x, y): get_neighbors(x, y, N)
        for x in range(N) for y in range(N)}
        
        results[size] = {}
        # loop over all ratios 
        for p_growth in p:            
            for d_death in d:
                ratio = p_growth / d_death
                tumor_sizes = []
                print(f' running for ratio {ratio}')
                
                for _ in range(k):  # Run k times
                    # Initialize grid 
                    grid = np.zeros((size, size), dtype=int)
                    center = size // 2
                    grid[center, center] = 1  # Initial tumor cell
                    # Simulate growth over t time steps 
                    for _ in range(T):
                        new_grid = grid.copy()
                        
                        for x, y in np.random.permutation([(i, j) for i in range(size) for j in range(size)]):
                            assert grid[x, y] in [0, 1, 2], f"Unexpected cell value {grid[x, y]} at ({x}, {y})"
                            if grid[x, y] == 1:
                                
                                for nx, ny in neighbors_map[(x, y)]:
                                    
                                    # If neighbors are healthy
                                    if grid[nx, ny] == 0 and np.random.rand() < p:
                                        new_grid[nx, ny] = 1 
                                
                            # Tumor cell death 
                            if grid[x, y] == 1 and np.random.rand() < d_death:
                                new_grid[x, y] = 2

                        grid = new_grid.copy()
                    
                    # Calculate total tumor size in the system after T 
                    tumor_size = np.sum(grid == 1)
                    tumor_sizes.append(tumor_size)

                # Store mean and standard deviation
                results[size][ratio] = (np.mean(tumor_sizes), np.std(tumor_sizes))

    return results


def sigmoid(x, A, B, C, D):
    """
    Computes the sigmoid function.

    The sigmoid function is defined as:
    A + (B - A) / (1 + (x / C) ** D)

    Parameters:
    x (float): The input value.
    A (float): The minimum value of the sigmoid function.
    B (float): The maximum value of the sigmoid function.
    C (float): The value of x at the midpoint of the sigmoid curve.
    D (float): The steepness of the sigmoid curve.

    Returns:
    float: The computed sigmoid value.
    """
    return A + (B - A) / (1 + (x / C) ** D)

def sigmoid_derivative(x, A, B, C, D):
    """
    Calculate the derivative of a sigmoid function.

    Parameters:
    x (float): The input value.
    A (float): The lower asymptote.
    B (float): The upper asymptote.
    C (float): The inflection point.
    D (float): The steepness of the curve.

    Returns:
    float: The derivative of the sigmoid function at x.
    """
    return - (B - A) * (D * (x / C) ** (D - 1)) / (C * (1 + (x / C) ** D) ** 2)

def plotter_peaks(data): 
    """
    Plots the fitted curves and their derivatives for tumor size vs growth/death ratio for different system sizes.
    Parameters:
    data (dict): A dictionary where keys are system sizes and values are dictionaries with growth/death ratios as keys 
                 and corresponding tumor sizes as values.
    The function performs the following:
    1. Plots the fitted sigmoid curves for tumor size vs growth/death ratio.
    2. Plots the derivatives of the fitted sigmoid curves.
    3. Plots the x-coordinate of the peak derivative vs grid size.
    The plots are displayed using matplotlib.
    """
    x_vals = np.linspace(0, 1300, 10000)
    plt.figure(figsize=(12, 6))
    for size, points in data.items():
        x = np.array(list(points.keys()))
        y = np.array([p[0] for p in points.values()])
        popt, _ = curve_fit(sigmoid, x, y, maxfev=5000)
        plt.plot(x_vals, sigmoid(x_vals, *popt), label=f'Fitted {size}')
        assert len(x) > 0, "x array is empty"
        assert len(y) > 0, "y array is empty"
        assert len(x) == len(y), "x and y arrays must have the same length"
    plt.legend(title = "Gris Size")
    plt.xlabel('Growth/Deat Ratio')
    plt.title('Fitted curves of tumor size (T=100) vs Growth/Death ratio for different system sizes ')
    plt.ylabel('Tumor size at T=100')
    plt.xscale('log')
    plt.grid()
    plt.show()

    peak_x_values = []
    plt.figure(figsize=(12, 6))
    for size, points in data.items():
        x = np.array(list(points.keys()))
        y = np.array([p[0] for p in points.values()])
        popt, _ = curve_fit(sigmoid, x, y, maxfev=5000)
        plt.plot(x_vals, sigmoid_derivative(x_vals, *popt), label=f'Derivative {size}')
        peak = np.max(sigmoid_derivative(x_vals, *popt))
        x_peak = x_vals[np.argmax(sigmoid_derivative(x_vals, *popt))]
        peak_x_values.append(x_peak)


    plt.legend()
    #plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Growth/Deat Ratio')
    plt.ylabel("(Tumor size at T=100)'")
    plt.title('Derivatives of Fitted curves of tumor size (T=100) vs Growth/Death ratio for different system sizes')
    plt.grid()
    plt.show()

    grid_sizes = list(data.keys())
    plt.figure(figsize=(12, 6))
    plt.plot(grid_sizes, peak_x_values, marker='o', linestyle='-')
    plt.xlabel('Grid Size')
    plt.ylabel('X Coordinate of Peak Derivative')
    plt.axhline(1, linestyle = "--", label = 'x = 1')
    plt.title('Peak Derivative X Coordinate vs Grid Size')
    plt.grid()
    plt.legend()

    plt.show()


# Run the simulation with averaging
if __name__ == '__main__':
    args = parse_args_peaks()
    N = args.GRID_SIZE
    T = args.TIME_STEPS
    p = args.GROWTH_PROBABILITY
    d = args.DEATH_PROBABILITY
    m = args.MUTATION_PROBABILITY

    k = 20  # Number of runs per configuration
    
    """
    results = simulate_growth(N, T, p, d, m, k)

    # Save results
    with open('averaged_results_finalk.pkl', 'wb') as f:
        pickle.dump(results, f)
    """
    # Load and plot results
    with open('averaged_results_finalk.pkl', 'rb') as f:
        loaded_results = pickle.load(f)

    #print(loaded_results)
    plotter_peaks(loaded_results)


    

