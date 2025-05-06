import numpy as np
import time
from .base import BaseSolver
from settings import DISCRETE_ACO_SETTINGS, PROGRESS_LOG_FREQUENCY

class DiscreteACO(BaseSolver):
    def __init__(self, tsp, 
                 num_ants=None, 
                 alpha=None, 
                 beta=None, 
                 rho=None, 
                 q=None, 
                 max_iterations=None, 
                 seed=None):
        """
        Initialize the Discrete ACO solver.
        
        Args:
            tsp: The TSP problem instance to solve
            num_ants: Number of ants
            alpha: Pheromone importance
            beta: Heuristic importance
            rho: Evaporation rate
            q: Pheromone deposit factor
            max_iterations: Maximum number of iterations
            seed: Random seed for reproducibility
        """
        super().__init__(tsp)
        
        # Use settings if parameters are not provided
        self.num_ants = num_ants if num_ants is not None else DISCRETE_ACO_SETTINGS['num_ants']
        self.alpha = alpha if alpha is not None else DISCRETE_ACO_SETTINGS['alpha']
        self.beta = beta if beta is not None else DISCRETE_ACO_SETTINGS['beta']
        self.rho = rho if rho is not None else DISCRETE_ACO_SETTINGS['rho']
        self.q = q if q is not None else DISCRETE_ACO_SETTINGS['q']
        self.max_iterations = max_iterations if max_iterations is not None else DISCRETE_ACO_SETTINGS['max_iterations']
        self.seed = seed if seed is not None else DISCRETE_ACO_SETTINGS['seed']
        
        # Set random seed
        np.random.seed(self.seed)
        
        # Initialize pheromone matrix
        self.num_cities = tsp.num_cities
        self.pheromone = np.ones((self.num_cities, self.num_cities))
        
        # Initialize heuristic information (inverse of distance)
        self.heuristic = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    self.heuristic[i][j] = 1.0 / tsp.distance_matrix[i][j]
    
    def solve(self):
        """Solve the TSP problem using Discrete Ant Colony Optimization."""
        start_time = time.time()
        
        # Reset history and best solution
        self.history = []
        self.best_path = None
        self.best_distance = float('inf')
        
        for iteration in range(self.max_iterations):
            # Path construction for each ant
            paths = []
            distances = []
            
            for ant in range(self.num_ants):
                path = self._construct_path()
                distance = self.tsp.get_total_distance(path)
                paths.append(path)
                distances.append(distance)
                
                # Update best solution if better
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_path = path.copy()
            
            # Update pheromones
            self._update_pheromones(paths, distances)
            
            # Record best distance for this iteration
            self.history.append(self.best_distance)
            
            # Print progress
            if (iteration + 1) % PROGRESS_LOG_FREQUENCY == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Best Distance: {self.best_distance:.2f}")
        
        self.execution_time = time.time() - start_time
        print(f"\nDiscrete ACO completed in {self.execution_time:.2f} seconds")
        print(f"Best Distance: {self.best_distance:.2f}")
        
        return self.best_path, self.best_distance
    
    def _construct_path(self):
        """Construct a path for an ant using pheromone and heuristic information."""
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
                pheromone_factor = self.pheromone[current_city][city] ** self.alpha
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
    
    def _update_pheromones(self, paths, distances):
        """Update pheromone levels based on ant paths."""
        # Evaporation
        self.pheromone *= (1 - self.rho)
        
        # Deposit new pheromones
        for path, distance in zip(paths, distances):
            # Determine the amount of pheromone to deposit
            deposit = self.q / distance
            
            # Update pheromones on each edge of the path
            for i in range(len(path) - 1):
                self.pheromone[path[i]][path[i+1]] += deposit
                self.pheromone[path[i+1]][path[i]] += deposit  # Symmetric update
            
            # Update pheromones on the edge connecting the last and first cities
            self.pheromone[path[-1]][path[0]] += deposit
            self.pheromone[path[0]][path[-1]] += deposit