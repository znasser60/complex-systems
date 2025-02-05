import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from tumor_growth_with_oxygen import parse_args, initialize_grid, get_neighbors, count_neighbors, update_oxygen

def analyze_clusters(cell_grid):
    """
    Analyze cluster sizes for normal cells, cancer cells, and quiescent cells.
    Returns cluster size distributions and largest cluster fractions (P∞) for each cell type.
    """
    cell_types = {1: "Normal", 2: "Cancer", 3: "Quiescent"}
    cluster_stats = {}
    
    for cell_type, name in cell_types.items():
        binary_grid = (cell_grid == cell_type).astype(int)  
        labeled_grid, num_clusters = label(binary_grid)  
        cluster_sizes = np.bincount(labeled_grid.ravel())[1:]  
        P_inf = cluster_sizes.max() / cell_grid.size if cluster_sizes.size > 0 else 0  
      
        cluster_stats[name] = {
            "cluster_sizes": cluster_sizes,
            "P_inf": P_inf,
            "num_clusters": num_clusters
        }
    
    return cluster_stats

def plot_clusters(cluster_data, time_steps):
    """
    Plot the largest cluster fraction for each cell type over time with shaded error areas. 
    """

    _, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = {
        "Normal": "#F28C8C",       
        "Cancer": "#A83232",       
        "Quiescent": "#4A0F0F"     
    }
    
    cell_types = ["Normal", "Cancer", "Quiescent"]
    
    for i, (oxygen_level, data) in enumerate(cluster_data.items()):
        ax = axes[i]
        for cell_type in cell_types:
            runs_matrix = np.array(data[cell_type])

            mean_P_inf = np.mean(runs_matrix, axis=0)
            std_P_inf = np.std(runs_matrix, axis=0)

            ax.plot(
                time_steps, mean_P_inf,
                label=f'{cell_type}',
                color=colors[cell_type],
                linestyle='-', linewidth=2  
            )

            ax.fill_between(
                time_steps,
                np.maximum(mean_P_inf - std_P_inf, 0),
                mean_P_inf + std_P_inf,
                color=colors[cell_type],
                alpha=0.3  
            )
            
        ax.set_xlabel("Time Step")
        ax.set_ylabel("P∞ (Largest Cluster Fraction)")
        ax.set_title(f'Oxygen Level = {oxygen_level}')
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=10, loc='upper right')
    
    plt.tight_layout()
    plt.show()


def simulate_cluster_growth(args, NUM_RUNS=10):
    '''
    Runs multiple simulations to compute mean & std dev of largest cluster fractions (P∞).
    '''
    N = args.GRID_SIZE
    T = args.TIME_STEPS
    oxygen_levels = [3.5e-4, 1.5e-2, 1.0e-1]  
    
    cluster_data = {oxygen_level: {cell_type: [] for cell_type in ["Normal", "Cancer", "Quiescent"]} 
                  for oxygen_level in oxygen_levels}
    
    for run in range(NUM_RUNS):  
        for oxygen in oxygen_levels: 
            args.INITIAL_OXYGEN = oxygen  
            cell_grid, oxygen_grid = initialize_grid(N, args)
            quiescent_clock = np.zeros((N, N), dtype=int)
            
            run_results = {cell_type: [] for cell_type in ["Normal", "Cancer", "Quiescent"]}  
            
            for step in range(T):
                new_grid = cell_grid.copy()
                for x in range(N):
                    for y in range(N):
                        if cell_grid[x, y] == 1:  # Normal cell
                            normal_cell_neighbors, cancer_cell_neighbors = count_neighbors(x, y, cell_grid, N)
                            local_oxygen_threshold = args.NT1 if normal_cell_neighbors > cancer_cell_neighbors else args.NT2
                            if oxygen_grid[x, y] < local_oxygen_threshold:  
                                new_grid[x, y] = 0
                            else:  # Attempts to divide
                                neighbors = get_neighbors(x, y, N)
                                empty_neighbors = [(nx, ny) for nx, ny in neighbors if cell_grid[nx, ny] == 0]
                                if empty_neighbors and np.random.rand() < args.GROWTH_PROBABILITY:
                                    nx, ny = empty_neighbors[np.random.choice(len(empty_neighbors))]
                                    new_grid[nx, ny] = 1
                        elif cell_grid[x, y] == 2:  # Cancer cell
                            normal_cell_neighbors, cancer_cell_neighbors = count_neighbors(x, y, cell_grid, N)
                            local_oxygen_threshold = args.CT1 if cancer_cell_neighbors > normal_cell_neighbors else args.CT2
                            if oxygen_grid[x, y] < local_oxygen_threshold: 
                                new_grid[x, y] = 3
                            else:  # Attempts to divide
                                neighbors = get_neighbors(x, y, N)
                                empty_neighbors = [(nx, ny) for nx, ny in neighbors if cell_grid[nx, ny] == 0]
                                if empty_neighbors and np.random.rand() < args.GROWTH_PROBABILITY:
                                    nx, ny = empty_neighbors[np.random.choice(len(empty_neighbors))]
                                    new_grid[nx, ny] = 2
                        elif cell_grid[x, y] == 3:  # Quiescent cancer cell
                            normal_cell_neighbors, cancer_cell_neighbors = count_neighbors(x, y, cell_grid, N)
                            if oxygen_grid[x, y] > args.CT1: 
                                new_grid[x, y] = 2
                                quiescent_clock[x, y] = 0
                            else: 
                                quiescent_clock[x, y] += 1
                                if quiescent_clock[x, y] > args.QUIESCENCE_TIME:  
                                    new_grid[x, y] = 0
                
                cell_grid = new_grid
                oxygen_grid = update_oxygen(cell_grid, oxygen_grid, args)

                if step % 5 == 0:  
                    cluster_stats = analyze_clusters(cell_grid)
                    for cell_type, stats in cluster_stats.items():
                        run_results[cell_type].append(stats["P_inf"])
            
            for cell_type in ["Normal", "Cancer", "Quiescent"]:
                cluster_data[oxygen][cell_type].append(run_results[cell_type])

    time_steps = range(0, T, 5)  
    plot_clusters(cluster_data, time_steps)

if __name__ == '__main__':
    args = parse_args()
    simulate_cluster_growth(args)
