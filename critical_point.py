import numpy as np
import matplotlib.pyplot as plt
import tumor_growth as tg
from argparse import ArgumentParser, RawTextHelpFormatter

def parse_args():
    "Parses inputs from commandline and returns them as a Namespace object."

    parser = ArgumentParser(prog = 'python3 critical_point.py',
        formatter_class = RawTextHelpFormatter, description =
        '  Simulate critical tumor size given arguments for growth probability, grid size and number of time steps.\n\n'
        '  Example syntax:\n'
        '    python3 critical_point.py -N 50 -T 100 -Np 30\n')

    # Optionals
    parser.add_argument('-N', dest='GRID_SIZE', type=int, default=50,
        help='grid size (default = 50)')

    parser.add_argument('-T', dest='TIME_STEPS', type=int, default=50,
        help='number of time steps (default = 100)')

    parser.add_argument('-Np', dest='N_GROWTH_PROBABILITIES', type=int, default=20,
        help='Number of values assumed for growth probability, equally spaced within the range between 0 and 1 (default = 20) ')

    parser.add_argument('-Nd', dest='N_DEATH_PROBABILITIES', type=int, default=20,
                        help='number of values assumed for death probability,equally spaced within range of 0 and 1, (default = 20) ')

    return parser.parse_args()


def plot(ratios, tumor_sizes, death_sizes, Np, Nd, T, N):
    # Plotting the results
    plt.scatter(ratios, tumor_sizes, marker='o', color='green', label="tumor")
    plt.scatter(ratios, death_sizes, marker='x', color='black', label="dead")
    plt.xscale('log')
    plt.axvline(x=1, color='red', linestyle='--', label='Critical Point (p = d)')
    plt.xlabel('Growth-to-Death Probability Ratio (p/d)')
    plt.ylabel('Final Tumor Size (% of grid size)')
    plt.title("Number of Dead and Tumorous Cells vs. Growth-to-Death Probability Ratio \n"
              f"(T = {T})")
    plt.grid(True)
    plt.legend(loc="center left")
    plt.savefig(f"data/Tumorsize_ratio_Np{Np}_Nd{Nd}_T{T}_N{N}_log_with_death.png")


if __name__ == '__main__':
    # Parameters
    args = parse_args()
    # Parameters
    N = args.GRID_SIZE  # Size of the grid (default 50x50)
    T = args.TIME_STEPS  # Number of simulation steps
    Np = args.N_GROWTH_PROBABILITIES  # Probability of tumor cell division
    Nd = args.N_DEATH_PROBABILITIES

    growth_probabilities = np.linspace(0.01, 0.99, Np)  # Growth probabilities from 0 to 1
    death_probabilities = np.linspace(0.01, 0.99, Nd)
    # Results storage
    ratios = []
    tumor_sizes = []  # Store final tumor size for each growth probability
    death_sizes = []

    # Run simulations for each growth probability
    for p in growth_probabilities:
        print(f"Growth probability: {p}")
        for d in death_probabilities:
            print(f"ratio p/d = {np.round(p/d, 3)}")
            grid = tg.simulate_growth(N, T, p,d)
            size = np.sum(grid == 1)
            death_size = np.sum(grid == 2)
            ratios.append(p/d)
            tumor_sizes.append(size/(N*N))
            death_sizes.append(death_size/(N*N))

    plot(ratios, tumor_sizes, death_sizes, Np, Nd, T, N)
    # TODO: plot average tumor size, then plot a curve through this, take the derivative of that and plot
    # the derivative against the ratio


