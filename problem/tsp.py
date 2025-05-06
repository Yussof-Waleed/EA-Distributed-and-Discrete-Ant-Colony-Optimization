import numpy as np
from .city import City
from settings import TSP_SETTINGS

class TSP:
    def __init__(self, num_cities=None, width=None, height=None, seed=None):
        """
        Initialize a TSP problem with a given number of cities randomly placed on a grid.
        
        Args:
            num_cities: Number of cities
            width: Width of the grid
            height: Height of the grid
            seed: Random seed for reproducibility
        """
        # Use settings if parameters are not provided
        self.num_cities = num_cities if num_cities is not None else TSP_SETTINGS['num_cities']
        self.width = width if width is not None else TSP_SETTINGS['width']
        self.height = height if height is not None else TSP_SETTINGS['height']
        self.seed = seed if seed is not None else TSP_SETTINGS['seed']
        
        # Set random seed for reproducibility
        np.random.seed(self.seed)
        
        # Generate cities with random coordinates
        self.cities = []
        for i in range(self.num_cities):
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)
            self.cities.append(City(x, y, id=i))
        
        # Calculate distance matrix
        self.distance_matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    self.distance_matrix[i][j] = self.cities[i].distance(self.cities[j])
    
    def get_distance(self, city1_idx, city2_idx):
        """Get the distance between two cities by their indices."""
        return self.distance_matrix[city1_idx][city2_idx]
    
    def get_total_distance(self, path):
        """Calculate the total distance of a path (a list of city indices)."""
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += self.get_distance(path[i], path[i+1])
        # Add distance from last city back to first city to complete the tour
        total_distance += self.get_distance(path[-1], path[0])
        return total_distance
    
    def __str__(self):
        """String representation of the TSP problem."""
        return f"TSP Problem with {self.num_cities} cities on a {self.width}x{self.height} grid"