"""
Discrete Ant Colony Optimization (DACO) for Traveling Salesman Problem

This implementation uses a discrete representation for the TSP, focusing on decision variables
(which city to visit next) rather than continuous parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import time


# ============== PROBLEM REPRESENTATION ==============

class City:
    """Class representing a city with x,y coordinates"""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Ant:
    """Class representing an ant in the colony"""
    
    def __init__(self):
        """Initialize an ant with empty tour and zero cost"""
        self.clear()
        
    def clear(self):
        """Reset the ant's state"""
        self.tour = []          # List of city indices representing the tour
        self.cost = 0.0         # Total tour distance


# ============== DISCRETE ACO ALGORITHM ==============

class DiscreteACO:
    """Discrete Ant Colony Optimization algorithm for TSP"""
    
    def __init__(self, cities, distance_function=None, num_ants=50, 
                 alpha=1.0, beta=2.0, evaporation_rate=0.1, q=100.0,
                 initial_pheromone=1.0):
        """Initialize the Discrete ACO algorithm
        
        Args:
            cities: List of city objects
            distance_function: Function to calculate distance between cities (if None, uses Euclidean)
            num_ants: Number of ants in the colony
            alpha: Importance of pheromone trail (α >= 0)
            beta: Importance of heuristic information (β >= 1)
            evaporation_rate: Pheromone evaporation rate (0 < ρ <= 1)
            q: Pheromone deposit factor
            initial_pheromone: Initial pheromone level on all edges
        """
        self.cities = cities[:]
        self.num_cities = len(cities)
        
        # If no distance function provided, use Euclidean distance
        if distance_function is None:
            self.distance_function = lambda c1, c2: np.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
        else:
            self.distance_function = distance_function
            
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q = q
        
        # Initialize ant colony
        self.ants = [Ant() for _ in range(num_ants)]
        
        # Initialize pheromone matrix with initial values
        self.pheromone = np.ones((self.num_cities, self.num_cities)) * initial_pheromone
        
        # Track best solution found so far
        self.best_ant = None
        self.best_cost = float('inf')
        
    def update(self):
        """Run one iteration of the algorithm"""
        # Let all ants construct their solutions
        self._construct_solutions()
        
        # Update pheromones
        self._update_pheromones()
        
        # Update best solution
        self._update_best_solution()
    
    def _construct_solutions(self):
        """Let all ants construct their solutions"""
        # Reset all ants
        for ant in self.ants:
            ant.clear()
            
            # Create list of unvisited cities
            unvisited = list(range(self.num_cities))
            
            # Select random starting city
            start_city = random.choice(unvisited)
            ant.tour.append(start_city)
            unvisited.remove(start_city)
            
            # Construct the rest of the tour
            current_city = start_city
            while unvisited:
                # Calculate probabilities for all unvisited cities
                probs = []
                for next_city in unvisited:
                    # Calculate attractiveness using pheromone and heuristic information
                    pheromone = self.pheromone[current_city][next_city] ** self.alpha
                    # Heuristic is inversely proportional to distance
                    heuristic = (1.0 / self.distance_function(self.cities[current_city], 
                                                         self.cities[next_city])) ** self.beta
                    probs.append(pheromone * heuristic)
                
                # Normalize probabilities
                probs = np.array(probs)
                prob_sum = np.sum(probs)
                if prob_sum > 0:
                    probs = probs / prob_sum
                else:
                    # If all pheromones are zero, use equal probability
                    probs = np.ones(len(unvisited)) / len(unvisited)
                
                # Select next city
                next_index = np.random.choice(len(unvisited), p=probs)
                next_city = unvisited[next_index]
                
                # Add to tour and update current city
                ant.tour.append(next_city)
                current_city = next_city
                unvisited.remove(next_city)
            
            # Calculate tour cost
            for i in range(len(ant.tour) - 1):
                ant.cost += self.distance_function(self.cities[ant.tour[i]], 
                                              self.cities[ant.tour[i + 1]])
            
            # Add cost to return to starting city (complete the loop)
            ant.cost += self.distance_function(self.cities[ant.tour[-1]], 
                                          self.cities[ant.tour[0]])
    
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
                self.pheromone[city_i][city_j] += deposit
                self.pheromone[city_j][city_i] += deposit  # Symmetric pheromone update
            
            # Close the tour loop
            city_i = ant.tour[-1]
            city_j = ant.tour[0]
            self.pheromone[city_i][city_j] += deposit
            self.pheromone[city_j][city_i] += deposit
    
    def _update_best_solution(self):
        """Update the best solution found so far"""
        for ant in self.ants:
            if self.best_ant is None or ant.cost < self.best_cost:
                self.best_ant = ant
                self.best_cost = ant.cost
    
    def get_best(self, num=1):
        """Get the best solutions found so far"""
        sorted_ants = sorted(self.ants, key=lambda a: a.cost)
        return sorted_ants[:num]
    
    def replace_worst(self, children_tours):
        """Replace worst ants with new solutions (for hybrid approach)"""
        # Calculate costs for child tours
        child_ants = []
        for tour in children_tours:
            new_ant = Ant()
            new_ant.tour = tour
            
            # Calculate tour cost
            for i in range(len(tour) - 1):
                new_ant.cost += self.distance_function(self.cities[tour[i]], self.cities[tour[i + 1]])
            new_ant.cost += self.distance_function(self.cities[tour[-1]], self.cities[tour[0]])
            
            child_ants.append(new_ant)
        
        # Sort ants by cost (descending) to find the worst ones
        self.ants.sort(key=lambda a: a.cost, reverse=True)
        
        # Replace the worst ants with children
        for i, child in enumerate(child_ants):
            if i < len(self.ants):  # Safety check
                self.ants[i] = child
                
                # Update best solution if needed
                if child.cost < self.best_cost:
                    self.best_ant = child
                    self.best_cost = child.cost
        
        # Re-sort the ants to maintain internal order expectations if needed
        self.ants.sort(key=lambda a: a.cost)


# ============== GENETIC OPERATIONS ==============

def order_crossover(parent1, parent2):
    """Order crossover operator for permutation encoding"""
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    
    # Initialize child with placeholder values
    child = [-1] * size
    
    # Copy segment from parent1
    child[start:end+1] = parent1[start:end+1]
    
    # Fill in remaining positions with cities from parent2
    ptr = (end + 1) % size
    for city in parent2:
        if city not in child:
            child[ptr] = city
            ptr = (ptr + 1) % size
    
    return child

def mutate(tour, mutation_rate=0.1):
    """Swap mutation operator"""
    tour = tour[:]
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]
    return tour

def generate_children(top_ants, num_children, mutation_rate=0.1):
    """Generate children from top performing ants"""
    children = []
    
    while len(children) < num_children:
        parent1 = random.choice(top_ants).tour
        parent2 = random.choice(top_ants).tour
        
        if parent1 != parent2:
            child_tour = order_crossover(parent1, parent2)
            child_tour = mutate(child_tour, mutation_rate)
            children.append(child_tour)
    
    return children


# ============== VISUALIZATION ==============

def visualize_solution(cities, tour, cost, history):
    """Visualize the TSP solution"""
    # Create figure with two subplots
    plt.figure(figsize=(12, 6))
    
    # Plot the tour
    plt.subplot(1, 2, 1)
    
    # Extract city coordinates from the tour
    x = [cities[city_idx].x for city_idx in tour]
    y = [cities[city_idx].y for city_idx in tour]
    
    # Add the starting city at the end to close the loop
    x.append(cities[tour[0]].x)
    y.append(cities[tour[0]].y)
    
    # Plot cities and tour
    plt.plot(x, y, 'ro-')
    plt.title(f'Best Tour Found by Distributed_ACO (Cost: {cost:.2f})')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Plot convergence history
    plt.subplot(1, 2, 2)
    plt.plot(range(len(history)), history, 'b-')
    plt.title('Total Distance Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Total Distance')
    
    plt.tight_layout()
    plt.savefig('assets/distributed_aco_tsp_solution.png', dpi=300)
    plt.show()


# ============== MAIN FUNCTION ==============

def main(seed=None):
    """Example program that uses Discrete ACO Algorithm for TSP"""
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Problem configuration
    n_cities = 50
    cities = [City(random.randint(0, 500), random.randint(0, 500)) for _ in range(n_cities)]
    
    # Algorithm parameters
    num_ants = 50
    alpha = 1.0
    beta = 2.0
    evaporation_rate = 0.1
    q = 100.0
    iterations = 100
    
    # Create the DACO algorithm
    daco = DiscreteACO(
        cities=cities,
        num_ants=num_ants,
        alpha=alpha,
        beta=beta,
        evaporation_rate=evaporation_rate,
        q=q
    )
    
    # Run the algorithm
    history = np.zeros(iterations)
    
    # Start timing
    start_time = time.time()
    
    # Main loop
    for iteration in range(iterations):
        # Run one iteration
        daco.update()
        
        # Record best cost for history
        best_ant = daco.get_best(1)[0]
        history[iteration] = best_ant.cost
        
        print(f'Iteration {iteration+1:3d}/{iterations} - Best Distance: {best_ant.cost:.2f}')
    
    # Calculate runtime
    runtime = time.time() - start_time
    
    # Get final best solution
    best_ant = daco.get_best(1)[0]
    
    # Output results
    print(f'\nAlgorithm Time Taken: {runtime:.2f} seconds')
    print(f'Best Distance: {best_ant.cost:.2f}')
    print(f'Best Tour: {[int(city) for city in best_ant.tour]}')
    
    # Visualize the results
    visualize_solution(cities, best_ant.tour, best_ant.cost, history)


if __name__ == '__main__':
    main(seed=42)