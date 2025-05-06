# General parameters with recommendation notes
SEED = 42  # Random seed for reproducibility

# Maximum number of iterations - controls runtime and solution quality
# More iterations generally yield better solutions but increase runtime
MAX_ITERATIONS = 100

# Frequency of logging progress
PROGRESS_LOG_FREQUENCY = 2

# Pheromone importance (Recommended range: 0.5 to 2.0)
# Higher values (>1): More exploitation of known good paths, risk of premature convergence
# Lower values (<1): More exploration, potentially slower convergence
# Optimal value: ~1.0 for balanced exploration/exploitation
ALPHA = 1.2

# Heuristic importance (Recommended range: 1.0 to 5.0)
# Higher values (>2): Greedier behavior, preferring shorter paths regardless of pheromones
# Lower values (<2): Less greedy, more influenced by pheromones
# Optimal value: ~2.0-3.0 for TSP problems
BETA = 2.3

# Evaporation rate (Recommended range: 0.1 to 0.9)
# Higher values (>0.5): Faster forgetting, helps avoid local optima but may slow convergence
# Lower values (<0.5): Slower forgetting, faster convergence but risk of premature convergence
# Optimal value: ~0.5 for balanced forgetting rate
RHO = 0.55

# Pheromone deposit factor (Recommended range: 10 to 1000, depends on problem scale)
# Higher values: More weight to good solutions
# Lower values: More gradual pheromone updates
# Optimal value: Adjust based on average tour length
Q = 100.0

# Settings for TSP problem
TSP_SETTINGS = {
    'num_cities': 100,         # Number of cities
    'width': 1000,             # Width of the grid
    'height': 1000,            # Height of the grid
    'seed': SEED,              # Random seed for reproducibility
    'max_iterations': MAX_ITERATIONS,  # Maximum number of iterations for ACO algorithms
}

# Settings for Discrete ACO
DISCRETE_ACO_SETTINGS = {
    'num_ants': 400,            # Number of ants
    'alpha': ALPHA,            # Pheromone importance
    'beta': BETA,              # Heuristic importance
    'rho': RHO,                # Evaporation rate
    'q': Q,                    # Pheromone deposit factor
    'max_iterations': MAX_ITERATIONS,  # Maximum number of iterations
    'seed': SEED,              # Random seed for reproducibility
}

# Settings for Distributed ACO
DISTRIBUTED_ACO_SETTINGS = {
    # Number of colonies (Recommended range: 2 to 8)
    # More colonies increase diversity but require more resources
    # Optimal value: ~4 for balanced performance
    'num_colonies': 4,         
    
    # Number of ants per colony (total ants = num_colonies * ants_per_colony)
    'ants_per_colony': 100,     
    
    'alpha': ALPHA,            # Pheromone importance
    'beta': BETA,              # Heuristic importance
    'rho': RHO,                # Evaporation rate
    'q': Q,                    # Pheromone deposit factor
    
    # Exchange frequency (Recommended range: 5-20 iterations)
    # Higher frequency: More rapid information sharing but less independent exploration
    # Lower frequency: More independent exploration but slower information propagation
    # Optimal value: ~10 iterations for balanced performance
    'exchange_freq': 10,       
    
    # Strategy for information exchange: 'best' or 'random'
    # 'best': Share only best solutions - faster convergence but less diversity
    # 'random': Share random information - more diversity but potentially slower convergence
    'exchange_strategy': 'random', 
    
    'max_iterations': MAX_ITERATIONS,  # Maximum number of iterations
    'seed': MAX_ITERATIONS,            # Random seed for reproducibility
}