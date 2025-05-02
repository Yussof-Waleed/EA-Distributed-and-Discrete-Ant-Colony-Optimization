# Ant Colony Optimization Algorithms for TSP
## Documentation and Integration Guide

This documentation provides a comprehensive guide to the two Ant Colony Optimization (ACO) algorithms implemented to solve the Traveling Salesman Problem (TSP): Discrete ACO and Distributed ACO.

## Table of Contents

1. Introduction to ACO
2. Discrete Ant Colony Optimization
3. Distributed Ant Colony Optimization
4. Algorithm Comparison
5. Installation Guide
6. Integration Guide
7. Parameter Tuning

## Introduction to ACO

Ant Colony Optimization is a metaheuristic inspired by the foraging behavior of ants. Ants deposit pheromones as they travel, creating paths that other ants can follow. Paths with higher pheromone concentrations become more attractive, leading to the emergence of optimal routes over time.

### Key ACO Concepts:

- **Pheromone Trails**: Represents the desirability of a path based on past solutions
- **Heuristic Information**: Usually inverse of distance, represents the a priori attractiveness of a move
- **Probabilistic Decision Rule**: Combines pheromone and heuristic information to make decisions
- **Pheromone Evaporation**: Reduces all pheromone values over time to avoid premature convergence
- **Pheromone Reinforcement**: Increases pheromone on edges used by good solutions

## Discrete Ant Colony Optimization

Discrete ACO focuses on discrete decision variables (which city to visit next) for solving combinatorial optimization problems like TSP.

### Algorithm Overview

1. **Initialization**:
   - Create a set of cities with coordinates
   - Create a distance matrix between all cities
   - Initialize pheromone values on all edges
   - Create a colony of ants

2. **Solution Construction**:
   - Each ant constructs a tour by selecting cities one by one
   - City selection is based on a probabilistic rule that considers:
     - Pheromone levels (τᵢⱼ) on edges
     - Heuristic information (ηᵢⱼ = 1/dᵢⱼ)

3. **Pheromone Update**:
   - Evaporate pheromones from all edges: τᵢⱼ ← (1-ρ)τᵢⱼ
   - Deposit new pheromones based on solution quality: τᵢⱼ ← τᵢⱼ + Δτᵢⱼᵏ

4. **Termination**:
   - Repeat steps 2-3 for a specified number of iterations
   - Return the best solution found

### Key Parameters

| Parameter | Symbol | Description | Typical Values |
|-----------|--------|-------------|---------------|
| Alpha | α | Importance of pheromone trail | 1.0 |
| Beta | β | Importance of heuristic information | 2.0-5.0 |
| Evaporation Rate | ρ | Rate at which pheromones evaporate | 0.1-0.5 |
| Pheromone Deposit Factor | Q | Scales pheromone deposit amount | 100-500 |
| Number of Ants | m | Size of the ant colony | 10-100 |
| Initial Pheromone | τ₀ | Initial pheromone level | 0.1-1.0 |

### Mathematical Formulas

**Probability of selection:**
The probability of ant k selecting city j after city i is:

$$p_{ij}^k = \frac{[\tau_{ij}]^\alpha \cdot [\eta_{ij}]^\beta}{\sum_{l \in unvisited} [\tau_{il}]^\alpha \cdot [\eta_{il}]^\beta}$$

Where:
- τᵢⱼ is the pheromone level on edge (i,j)
- ηᵢⱼ is the heuristic value (1/dᵢⱼ)
- α and β control the relative importance of pheromone versus distance

**Pheromone update rule:**

$$\tau_{ij} \leftarrow (1-\rho) \cdot \tau_{ij} + \sum_{k=1}^{m} \Delta\tau_{ij}^k$$

Where:
- ρ is the evaporation rate
- Δτᵢⱼᵏ is the pheromone deposited by ant k on edge (i,j)

$$\Delta\tau_{ij}^k = \begin{cases} 
\frac{Q}{L_k} & \text{if ant k uses edge (i,j) in its tour} \\
0 & \text{otherwise}
\end{cases}$$

Where:
- Q is the pheromone deposit factor
- Lₖ is the length of the tour constructed by ant k

## Distributed Ant Colony Optimization

Distributed ACO enhances the basic ACO approach by incorporating parallel processing concepts and genetic algorithm techniques.

### Algorithm Overview

1. **Initialization**:
   - Similar to Discrete ACO, but operates directly on city objects
   - Emphasizes decentralized computation

2. **Solution Construction**:
   - Similar to Discrete ACO
   - Uses a functional approach for distance calculations

3. **Genetic Operations**:
   - Incorporates Order Crossover (OX) to create new solutions
   - Applies mutation to maintain diversity
   - Replaces worst solutions with genetically created offspring

4. **Pheromone Update and Termination**:
   - Similar to Discrete ACO

### Key Parameters

Same parameters as Discrete ACO, plus:

| Parameter | Description | Typical Values |
|-----------|-------------|---------------|
| Mutation Rate | Probability of mutation in genetic operations | 0.1-0.2 |
| Number of Children | Number of offspring to generate per iteration | 5-20 |

### Mathematical Formulas

Same as Discrete ACO, plus genetic operations:

**Order Crossover (OX):**
1. Select a random subsequence from parent 1
2. Copy that subsequence to the child solution
3. Fill the remaining positions with cities from parent 2 in the order they appear, skipping those already in the child

**Mutation:**
Swap two randomly selected cities in the tour with probability equal to the mutation rate.

## Algorithm Comparison

| Feature | Discrete ACO | Distributed ACO |
|---------|-------------|-----------------|
| **Problem Representation** | Uses distance matrix | Calculates distances on demand |
| **Ant Representation** | Includes visited set for faster lookup | Simpler representation |
| **City Selection** | Selection by roulette wheel | Similar approach |
| **Genetic Operations** | Not included | Includes crossover and mutation |
| **Memory Efficiency** | Pre-computes distances (faster but more memory) | Calculates distances as needed (less memory) |
| **Solution Quality** | Good for small to medium instances | Better for larger instances with genetic operations |
| **Parallelization** | Not designed for parallelization | Could be parallelized |
| **Implementation Complexity** | Moderate | Moderate to High |

## Installation Guide

### Prerequisites
- Python 3.6+
- NumPy
- Matplotlib

### Installation Steps

1. Install the required packages:

```bash
pip install numpy matplotlib
```

2. Download the source code:

```bash
git clone https://github.com/yourusername/ant-colony-optimization.git
cd ant-colony-optimization
```

3. Run the algorithms:

```bash
# For Discrete ACO
python discete_ACO.py

# For Distributed ACO
python distributed_ACO.py
```

## Integration Guide

To integrate both algorithms into a unified framework (such as a GUI application):

1. **Standardize Constants:**
   Both files use similar constants. Create a shared constants file:

```python
# constants.py
# Problem settings
DEFAULT_NUM_CITIES = 50
CITY_MIN_COORD = 0
CITY_MAX_COORD = 500

# Algorithm parameters
DEFAULT_NUM_ANTS = 50
DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 2.0
DEFAULT_EVAP_RATE = 0.1
DEFAULT_Q = 100.0
DEFAULT_INIT_PHEROMONE = 1.0
DEFAULT_MAX_ITERATIONS = 100

# Genetic algorithm parameters
DEFAULT_MUTATION_RATE = 0.1
DEFAULT_NUM_CHILDREN = 10

# Visualization settings
PLOT_FIGSIZE = (12, 6)
TOUR_PLOT_STYLE = 'ro-'
HISTORY_PLOT_STYLE = 'b-'

# Reporting settings
PRINT_FREQUENCY = 1
```

2. **Create a Common Interface:**
   Implement a base ACO class and have both algorithms inherit from it:

```python
# base_aco.py
class BaseACO:
    def __init__(self):
        self.best_cost = float('inf')
    
    def run(self, max_iterations):
        """Must be implemented by subclasses"""
        pass
    
    def get_best(self, num=1):
        """Must be implemented by subclasses"""
        pass
    
    def visualize_solution(self):
        """Must be implemented by subclasses"""
        pass
```

3. **Unify Problem Representation:**
   Create a shared TSP problem class:

```python
# tsp_problem.py
class TSPProblem:
    def __init__(self, num_cities=DEFAULT_NUM_CITIES, random_seed=None):
        # Implementation from discete_ACO.py
        pass
    
    def get_distance(self, i, j):
        # Return distance between cities
        pass
    
    def get_cities(self):
        # Return the list of cities
        pass
```

4. **API for GUI Integration:**
   Create a wrapper class that provides a unified API:

```python
# aco_solver.py
class ACOSolver:
    def __init__(self, algorithm_type="discrete", **kwargs):
        if algorithm_type == "discrete":
            self.algorithm = DiscreteACO(**kwargs)
        else:
            self.algorithm = DistributedACO(**kwargs)
    
    def solve(self, max_iterations=DEFAULT_MAX_ITERATIONS):
        return self.algorithm.run(max_iterations)
    
    def get_best_solution(self):
        return self.algorithm.get_best(1)[0]
    
    def visualize(self):
        self.algorithm.visualize_solution()
```

5. **Example Integration:**

```python
# example_gui_integration.py
def run_aco_from_gui(algorithm_type, num_cities, num_ants, alpha, beta, iterations):
    problem = TSPProblem(num_cities=num_cities)
    
    solver = ACOSolver(
        algorithm_type=algorithm_type,
        problem=problem,
        num_ants=num_ants,
        alpha=alpha,
        beta=beta
    )
    
    best_tour, best_cost, history = solver.solve(max_iterations=iterations)
    
    # Return results to GUI
    return {
        "best_tour": best_tour,
        "best_cost": best_cost,
        "history": history
    }
```

## Parameter Tuning

For optimal performance, consider these guidelines for tuning parameters:

1. **Alpha (α):**
   - Higher values (>1.0) increase the importance of pheromone trails
   - Lower values (<1.0) make the algorithm more greedy
   - Recommended range: 0.5 to 2.0

2. **Beta (β):**
   - Higher values increase the importance of distance
   - Recommended range: 2.0 to 5.0

3. **Evaporation Rate (ρ):**
   - Higher values (closer to 1.0) make the algorithm forget poor solutions faster
   - Lower values (closer to 0.0) increase algorithm stability but may lead to stagnation
   - Recommended range: 0.1 to 0.5

4. **Number of Ants:**
   - More ants explore more solutions but increase computation time
   - A good rule of thumb is to use the same number of ants as cities

5. **Mutation Rate (Distributed ACO only):**
   - Higher values increase diversity but may disrupt good solutions
   - Recommended range: 0.05 to 0.2

By adjusting these parameters, you can control the balance between exploration (finding new solutions) and exploitation (refining known good solutions).