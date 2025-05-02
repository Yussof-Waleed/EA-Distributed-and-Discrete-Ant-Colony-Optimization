"""
Discrete Ant Colony Optimization (DACO) for Traveling Salesman Problem

This implementation uses a discrete representation for the TSP, focusing on decision variables
(which city to visit next) rather than continuous parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import time


# ============== CONSTANTS ==============

# Problem settings
DEFAULT_NUM_CITIES = 50
CITY_MIN_COORD = 0
CITY_MAX_COORD = 500

# Algorithm parameters
DEFAULT_NUM_ANTS = 50
DEFAULT_ALPHA = 1.0             # Pheromone importance (α >= 0)
DEFAULT_BETA = 2.0              # Heuristic importance (β >= 1)
DEFAULT_EVAP_RATE = 0.1         # Pheromone evaporation rate (0 < ρ <= 1)
DEFAULT_Q = 100.0               # Pheromone deposit factor
DEFAULT_INIT_PHEROMONE = 1.0    # Initial pheromone level on all edges
DEFAULT_MAX_ITERATIONS = 100    # Default number of iterations to run

# Visualization settings
PLOT_FIGSIZE = (12, 6)
TOUR_PLOT_STYLE = 'ro-'         # Red circles with lines
HISTORY_PLOT_STYLE = 'b-'       # Blue line

# Reporting settings
PRINT_FREQUENCY = 1             # How often to print progress (iterations)


# ============== PROBLEM REPRESENTATION ==============

class City:
    """Class representing a city with x,y coordinates"""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y


class TSPProblem:
    """Class representing a TSP instance"""
    
    def __init__(self, num_cities: int = DEFAULT_NUM_CITIES, random_seed: int = None):
        """Initialize a TSP problem with randomly generated cities"""
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
            
        self.num_cities = num_cities
        # Generate random coordinates for cities in a grid
        self.cities = [City(random.randint(CITY_MIN_COORD, CITY_MAX_COORD), 
                           random.randint(CITY_MIN_COORD, CITY_MAX_COORD)) 
                      for _ in range(num_cities)]
        # Calculate distance matrix between all cities
        self.distance_matrix = self._compute_distance_matrix()
        
    def _compute_distance_matrix(self):
        """Compute the Euclidean distance matrix between all cities"""
        dist_matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    city_i = self.cities[i]
                    city_j = self.cities[j]
                    dist = np.sqrt((city_i.x - city_j.x)**2 + (city_i.y - city_j.y)**2)
                    dist_matrix[i, j] = dist
        return dist_matrix
    
    def get_distance(self, i: int, j: int) -> float:
        """Get the distance between two cities by their indices"""
        return self.distance_matrix[i, j]


# ============== ANT REPRESENTATION ==============

class Ant:
    """Class representing an ant in the colony"""
    
    def __init__(self, num_cities: int):
        """Initialize an ant with empty tour and zero cost"""
        self.num_cities = num_cities
        self.tour = []          # List of city indices representing the tour
        self.cost = 0.0         # Total tour distance
        self.visited = set()    # Set of visited cities for faster lookup
        
    def clear(self):
        """Reset the ant's state"""
        self.tour = []
        self.cost = 0.0
        self.visited = set()
        
    def visit_city(self, city_idx: int, distance: float):
        """Add a city to the ant's tour"""
        self.tour.append(city_idx)
        self.visited.add(city_idx)
        self.cost += distance
        
    def has_visited(self, city_idx: int) -> bool:
        """Check if the ant has visited a particular city"""
        return city_idx in self.visited


# ============== DISCRETE ACO ALGORITHM ==============

class DiscreteACO:
    """Discrete Ant Colony Optimization algorithm for TSP"""
    
    def __init__(self, problem: TSPProblem, num_ants: int = DEFAULT_NUM_ANTS, 
                 alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA, 
                 evaporation_rate: float = DEFAULT_EVAP_RATE, q: float = DEFAULT_Q,
                 initial_pheromone: float = DEFAULT_INIT_PHEROMONE):
        """Initialize the Discrete ACO algorithm
        
        Args:
            problem: TSP problem instance
            num_ants: Number of ants in the colony
            alpha: Importance of pheromone trail (α >= 0)
            beta: Importance of heuristic information (β >= 1)
            evaporation_rate: Pheromone evaporation rate (0 < ρ <= 1)
            q: Pheromone deposit factor
            initial_pheromone: Initial pheromone level on all edges
        """
        self.problem = problem
        self.num_cities = problem.num_cities
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q = q
        
        # Initialize ant colony
        self.ants = [Ant(self.num_cities) for _ in range(num_ants)]
        
        # Initialize pheromone matrix with initial values
        self.pheromone = np.ones((self.num_cities, self.num_cities)) * initial_pheromone
        
        # Track best solution found so far
        self.best_tour = None
        self.best_cost = float('inf')
        
    def _construct_solutions(self):
        """Let all ants construct their solutions"""
        # Reset all ants
        for ant in self.ants:
            ant.clear()
            
            # Select random starting city
            start_city = random.randint(0, self.num_cities - 1)
            ant.visit_city(start_city, 0.0)  # First city has no distance cost
            
            # Construct the rest of the tour
            for _ in range(self.num_cities - 1):
                current_city = ant.tour[-1]
                next_city = self._select_next_city(ant, current_city)
                distance = self.problem.get_distance(current_city, next_city)
                ant.visit_city(next_city, distance)
            
            # Complete the tour by returning to the starting city
            start_city = ant.tour[0]
            final_distance = self.problem.get_distance(ant.tour[-1], start_city)
            ant.cost += final_distance  # Add the return trip cost
    
    def _select_next_city(self, ant: Ant, current_city: int) -> int:
        """Select the next city for an ant using the ACO decision rule"""
        # Calculate probabilities for all unvisited cities
        probabilities = np.zeros(self.num_cities)
        denominator = 0.0
        
        # For each possible next city
        for city in range(self.num_cities):
            if not ant.has_visited(city):
                # Calculate attractiveness using pheromone and heuristic information
                pheromone = self.pheromone[current_city, city] ** self.alpha
                # Heuristic is inversely proportional to distance
                heuristic = (1.0 / self.problem.get_distance(current_city, city)) ** self.beta
                probabilities[city] = pheromone * heuristic
                denominator += probabilities[city]
        
        # Normalize probabilities
        if denominator > 0:
            probabilities = probabilities / denominator
        else:
            # If all pheromones are zero, use equal probability for all unvisited cities
            unvisited_cities = [c for c in range(self.num_cities) if not ant.has_visited(c)]
            for city in unvisited_cities:
                probabilities[city] = 1.0 / len(unvisited_cities)
        
        # Select next city using roulette wheel selection
        return np.random.choice(self.num_cities, p=probabilities)
    
    def _update_pheromones(self):
        """Update pheromone trails using the ant solutions"""
        # Evaporate all pheromones
        self.pheromone *= (1.0 - self.evaporation_rate)
        
        # Add new pheromones from each ant's tour
        for ant in self.ants:
            # Pheromone deposit is inversely proportional to tour cost
            deposit = self.q / ant.cost
            
            # Deposit pheromone on all edges in the ant's tour
            for i in range(len(ant.tour) - 1):
                city_i = ant.tour[i]
                city_j = ant.tour[i + 1]
                self.pheromone[city_i, city_j] += deposit
                self.pheromone[city_j, city_i] += deposit  # Symmetric pheromone update
            
            # Close the tour loop
            city_i = ant.tour[-1]
            city_j = ant.tour[0]
            self.pheromone[city_i, city_j] += deposit
            self.pheromone[city_j, city_i] += deposit
    
    def _update_best_solution(self):
        """Update the best solution found so far"""
        for ant in self.ants:
            if ant.cost < self.best_cost:
                self.best_cost = ant.cost
                self.best_tour = ant.tour.copy()
    
    def run(self, max_iterations: int = DEFAULT_MAX_ITERATIONS, verbose: bool = True):
        """Run the Discrete ACO algorithm
        
        Args:
            max_iterations: Maximum number of iterations
            verbose: Whether to print progress
            
        Returns:
            Tuple of (best_tour, best_cost)
        """
        # Initialize history to track progress
        history = np.zeros(max_iterations)
        
        # Start timing
        start_time = time.time()
        
        # Main ACO loop
        for iteration in range(max_iterations):
            # Let ants construct solutions
            self._construct_solutions()
            
            # Update best solution
            self._update_best_solution()
            
            # Update pheromones
            self._update_pheromones()
            
            # Record best cost for history
            history[iteration] = self.best_cost
            
            # Print progress
            if verbose and (iteration + 1) % PRINT_FREQUENCY == 0:
                print(f'Iteration {iteration+1:2d}/{max_iterations} - Best Distance: {self.best_cost:.2f}')
        
        # Calculate runtime
        runtime = time.time() - start_time
        
        if verbose:
            print(f'\nAlgorithm Time Taken: {runtime:.2f} seconds')
            print(f'Best Distance: {self.best_cost:.2f}')
            print(f'Best Tour: {[int(city) for city in self.best_tour]}')
        
        return self.best_tour, self.best_cost, history
    
    def get_best(self, num=1):
        """Get the best solutions found so far
        
        This mimics the interface of the HybridACO class for GUI integration
        """
        sorted_ants = sorted(self.ants, key=lambda a: a.cost)
        return sorted_ants[:num]


# ============== VISUALIZATION ==============

def visualize_solution(problem: TSPProblem, tour: list, cost: float, history):
    """Visualize the TSP solution"""
    # Create a figure with two subplots
    plt.figure(figsize=PLOT_FIGSIZE)
    
    # Plot the tour
    plt.subplot(1, 2, 1)
    
    # Extract city coordinates from the tour
    x = [problem.cities[city_idx].x for city_idx in tour]
    y = [problem.cities[city_idx].y for city_idx in tour]
    
    # Add the starting city at the end to close the loop
    x.append(problem.cities[tour[0]].x)
    y.append(problem.cities[tour[0]].y)
    
    # Plot cities and tour
    plt.plot(x, y, TOUR_PLOT_STYLE)
    plt.title(f'Best Tour Found (Cost: {cost:.2f})')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Plot convergence history
    plt.subplot(1, 2, 2)
    plt.plot(range(len(history)), history, HISTORY_PLOT_STYLE)
    plt.title('Total Distance Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Total Distance')
    
    plt.tight_layout()
    plt.show()


# ============== MAIN FUNCTION ==============

def main(seed=None):
    """Example program that uses Discrete ACO Algorithm for TSP"""
    # Problem configuration
    problem = TSPProblem(num_cities=DEFAULT_NUM_CITIES, random_seed=seed)
    
    # Create and run the algorithm
    aco = DiscreteACO(
        problem=problem, 
        num_ants=DEFAULT_NUM_ANTS,
        alpha=DEFAULT_ALPHA,
        beta=DEFAULT_BETA,
        evaporation_rate=DEFAULT_EVAP_RATE,
        q=DEFAULT_Q
    )
    
    # Run the algorithm
    best_tour, best_cost, history = aco.run(max_iterations=DEFAULT_MAX_ITERATIONS, verbose=True)
    
    # Visualize the results
    visualize_solution(problem, best_tour, best_cost, history)


if __name__ == '__main__':
    main(seed=42)