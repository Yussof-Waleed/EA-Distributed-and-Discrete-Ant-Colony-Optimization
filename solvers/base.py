import numpy as np
import matplotlib.pyplot as plt
import time
from abc import ABC, abstractmethod

class BaseSolver(ABC):
    def __init__(self, tsp):
        """
        Initialize the base solver with a TSP problem.
        
        Args:
            tsp: The TSP problem instance to solve
        """
        self.tsp = tsp
        self.best_path = None
        self.best_distance = float('inf')
        self.history = []  # To store the best distance at each iteration
        self.execution_time = 0
        
    @abstractmethod
    def solve(self):
        """Solve the TSP problem. Must be implemented by subclasses."""
        pass
    
    def plot_solution(self, filename=None):
        """
        Plot the best solution found.
        
        Args:
            filename: If provided, save the plot to this file
        """
        if self.best_path is None:
            print("No solution found yet. Run the solver first.")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot cities
        x = [self.tsp.cities[i].x for i in self.best_path]
        y = [self.tsp.cities[i].y for i in self.best_path]
        
        # Add the first city at the end to complete the tour
        x.append(self.tsp.cities[self.best_path[0]].x)
        y.append(self.tsp.cities[self.best_path[0]].y)
        
        plt.plot(x, y, 'o-', markersize=10)
        
        # Add city labels
        for i, city_idx in enumerate(self.best_path):
            plt.text(x[i], y[i], str(city_idx), fontsize=12)
        
        plt.title(f"Best Path (Distance: {self.best_distance:.2f})")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True)
        
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
            
        plt.close()
        
    def plot_convergence(self, filename=None):
        """
        Plot the convergence history.
        
        Args:
            filename: If provided, save the plot to this file
        """
        if not self.history:
            print("No history available. Run the solver first.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.history) + 1), self.history, 'b-')
        plt.title(f"Convergence History (Final Distance: {self.best_distance:.2f})")
        plt.xlabel("Iteration")
        plt.ylabel("Best Distance")
        plt.grid(True)
        
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
            
        plt.close()