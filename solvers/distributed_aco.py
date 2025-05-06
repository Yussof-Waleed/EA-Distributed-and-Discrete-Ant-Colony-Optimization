import numpy as np
import time
from .base import BaseSolver
from settings import DISTRIBUTED_ACO_SETTINGS, PROGRESS_LOG_FREQUENCY

class DistributedACO(BaseSolver):
    def __init__(self, tsp, 
                 num_colonies=None,
                 ants_per_colony=None, 
                 alpha=None, 
                 beta=None, 
                 rho=None, 
                 q=None,
                 exchange_freq=None,
                 exchange_strategy=None,
                 max_iterations=None, 
                 seed=None):
        """
        Initialize the Distributed ACO solver.
        
        Args:
            tsp: The TSP problem instance to solve
            num_colonies: Number of ant colonies
            ants_per_colony: Number of ants per colony
            alpha: Pheromone importance
            beta: Heuristic importance
            rho: Evaporation rate
            q: Pheromone deposit factor
            exchange_freq: Frequency of information exchange between colonies
            exchange_strategy: Strategy for information exchange: 'best', 'random'
            max_iterations: Maximum number of iterations
            seed: Random seed for reproducibility
        """
        super().__init__(tsp)
        
        # Use settings if parameters are not provided
        self.num_colonies = num_colonies if num_colonies is not None else DISTRIBUTED_ACO_SETTINGS['num_colonies']
        self.ants_per_colony = ants_per_colony if ants_per_colony is not None else DISTRIBUTED_ACO_SETTINGS['ants_per_colony']
        self.alpha = alpha if alpha is not None else DISTRIBUTED_ACO_SETTINGS['alpha']
        self.beta = beta if beta is not None else DISTRIBUTED_ACO_SETTINGS['beta']
        self.rho = rho if rho is not None else DISTRIBUTED_ACO_SETTINGS['rho']
        self.q = q if q is not None else DISTRIBUTED_ACO_SETTINGS['q']
        self.exchange_freq = exchange_freq if exchange_freq is not None else DISTRIBUTED_ACO_SETTINGS['exchange_freq']
        self.exchange_strategy = exchange_strategy if exchange_strategy is not None else DISTRIBUTED_ACO_SETTINGS['exchange_strategy']
        self.max_iterations = max_iterations if max_iterations is not None else DISTRIBUTED_ACO_SETTINGS['max_iterations']
        self.seed = seed if seed is not None else DISTRIBUTED_ACO_SETTINGS['seed']
        
        # Set random seed
        np.random.seed(self.seed)
        
        # Initialize colony-specific data
        self.num_cities = tsp.num_cities
        
        # Each colony has its own pheromone matrix
        self.pheromones = [np.ones((self.num_cities, self.num_cities)) for _ in range(self.num_colonies)]
        
        # Initialize heuristic information (inverse of distance) - shared across colonies
        self.heuristic = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    self.heuristic[i][j] = 1.0 / tsp.distance_matrix[i][j]
        
        # Track best solutions for each colony
        self.colony_best_paths = [None] * self.num_colonies
        self.colony_best_distances = [float('inf')] * self.num_colonies
    
    def solve(self):
        """Solve the TSP problem using Distributed Ant Colony Optimization."""
        start_time = time.time()
        
        # Reset history and best solution
        self.history = []
        self.best_path = None
        self.best_distance = float('inf')
        
        for iteration in range(self.max_iterations):
            # For each colony
            for colony in range(self.num_colonies):
                # Path construction for each ant in the colony
                paths = []
                distances = []
                
                for ant in range(self.ants_per_colony):
                    path = self._construct_path(colony)
                    distance = self.tsp.get_total_distance(path)
                    paths.append(path)
                    distances.append(distance)
                    
                    # Update colony's best solution
                    if distance < self.colony_best_distances[colony]:
                        self.colony_best_distances[colony] = distance
                        self.colony_best_paths[colony] = path.copy()
                        
                        # Update global best solution
                        if distance < self.best_distance:
                            self.best_distance = distance
                            self.best_path = path.copy()
                
                # Update pheromones for this colony
                self._update_pheromones(colony, paths, distances)
            
            # Information exchange between colonies
            if (iteration + 1) % self.exchange_freq == 0:
                self._exchange_information()
            
            # Record best distance for this iteration
            self.history.append(self.best_distance)
            
            # Print progress
            if (iteration + 1) % PROGRESS_LOG_FREQUENCY == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Best Distance: {self.best_distance:.2f}")
        
        self.execution_time = time.time() - start_time
        print(f"\nDistributed ACO completed in {self.execution_time:.2f} seconds")
        print(f"Best Distance: {self.best_distance:.2f}")
        
        return self.best_path, self.best_distance
    
    def _construct_path(self, colony):
        """Construct a path for an ant in the given colony."""
        # Start at a random city
        start_city = np.random.randint(0, self.num_cities)
        path = [start_city]
        unvisited = set(range(self.num_cities))
        unvisited.remove(start_city)
        
        # Construct path by selecting next city
        current_city = start_city
        while unvisited:
            # Calculate probabilities for each unvisited city
            probabilities = []
            for city in unvisited:
                pheromone_factor = self.pheromones[colony][current_city][city] ** self.alpha
                heuristic_factor = self.heuristic[current_city][city] ** self.beta
                probabilities.append(pheromone_factor * heuristic_factor)
            
            # Normalize probabilities
            probabilities = np.array(probabilities)
            if probabilities.sum() > 0:
                probabilities = probabilities / probabilities.sum()
            else:
                probabilities = np.ones(len(unvisited)) / len(unvisited)
            
            # Select next city using roulette wheel selection
            next_city_idx = np.random.choice(len(unvisited), p=probabilities)
            next_city = list(unvisited)[next_city_idx]
            
            # Add next city to path
            path.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city
        
        return path
    
    def _update_pheromones(self, colony, paths, distances):
        """Update pheromone levels for a colony based on ant paths."""
        # Evaporation
        self.pheromones[colony] *= (1 - self.rho)
        
        # Deposit new pheromones
        for path, distance in zip(paths, distances):
            # Determine the amount of pheromone to deposit
            deposit = self.q / distance
            
            # Update pheromones on each edge of the path
            for i in range(len(path) - 1):
                self.pheromones[colony][path[i]][path[i+1]] += deposit
                self.pheromones[colony][path[i+1]][path[i]] += deposit  # Symmetric update
            
            # Update pheromones on the edge connecting the last and first cities
            self.pheromones[colony][path[-1]][path[0]] += deposit
            self.pheromones[colony][path[0]][path[-1]] += deposit
    
    def _exchange_information(self):
        """Exchange information between colonies based on the selected strategy."""
        if self.exchange_strategy == 'best':
            # Find the colony with the best solution
            best_colony = np.argmin(self.colony_best_distances)
            best_path = self.colony_best_paths[best_colony]
            
            # Influence other colonies' pheromone matrices with the best solution
            for colony in range(self.num_colonies):
                if colony != best_colony:
                    # Blend some of the best colony's pheromones with this colony's
                    self.pheromones[colony] = 0.7 * self.pheromones[colony] + 0.3 * self.pheromones[best_colony]
                    
                    # Additionally, deposit pheromones on the best path
                    deposit = self.q / self.colony_best_distances[best_colony]
                    for i in range(len(best_path) - 1):
                        self.pheromones[colony][best_path[i]][best_path[i+1]] += deposit
                        self.pheromones[colony][best_path[i+1]][best_path[i]] += deposit
                    
                    self.pheromones[colony][best_path[-1]][best_path[0]] += deposit
                    self.pheromones[colony][best_path[0]][best_path[-1]] += deposit
                    
        elif self.exchange_strategy == 'random':
            # Each colony shares information with a random other colony
            for colony in range(self.num_colonies):
                # Select a random different colony
                other_colony = colony
                while other_colony == colony:
                    other_colony = np.random.randint(0, self.num_colonies)
                
                # Mix pheromones
                self.pheromones[colony] = 0.8 * self.pheromones[colony] + 0.2 * self.pheromones[other_colony]
                self.pheromones[other_colony] = 0.8 * self.pheromones[other_colony] + 0.2 * self.pheromones[colony]