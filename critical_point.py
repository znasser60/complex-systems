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

    parser.add_argument('-T', dest='TIME_STEPS', type=int, default=100,
        help='number of time steps (default = 100)')

    parser.add_argument('-Np', dest='N_GROWTH_PROBABILITIES', type=int, default=30,
        help='Number of values assumed for growth probability, equally spaced within the range between 0 and 1 (default = 30) ')

    parser.add_argument('-d', dest='DEATH_PROBABILITY', type=float, default=0.01,
                        help='death probability, probability that tumor cell dies, (default = 0.01) ')

    return parser.parse_args()


def plot(growth_probabilities, tumor_sizes, Np, d, T, N):
    # Plotting the results
    plt.plot(growth_probabilities, tumor_sizes, marker='o')
    #plt.xscale('log')
    plt.xlabel("Growth Probability (p)")
    plt.ylabel("Final Tumor Size")
    plt.title("Tumor Growth as a Function of Growth Probability \n"
              f"(d = {d}, T = {T})")
    plt.grid(True)
    plt.savefig(f"data/Tumorsize_Np{Np}_d{d}_T{T}_N{N}_nolog.png")


if __name__ == '__main__':
    # Parameters
    args = parse_args()
    # Parameters
    N = args.GRID_SIZE  # Size of the grid (default 50x50)
    T = args.TIME_STEPS  # Number of simulation steps
    Np = args.N_GROWTH_PROBABILITIES  # Probability of tumor cell division
    d = args.DEATH_PROBABILITY

    growth_probabilities = np.linspace(0.0, 1.0, Np)  # Growth probabilities from 0 to 1
    # Results storage
    tumor_sizes = []  # Store final tumor size for each growth probability

    # Run simulations for each growth probability
    for p in growth_probabilities:
        print(f"Growth probability: {p}")
        grid = tg.simulate_growth(N, T, p,d)
        size = np.sum(grid == 1)
        tumor_sizes.append(size)

    plot(growth_probabilities, tumor_sizes, Np, d, T, N)


