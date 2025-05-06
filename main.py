import matplotlib.pyplot as plt
from problem.tsp import TSP
from solvers.discrete_aco import DiscreteACO
from solvers.distributed_aco import DistributedACO
import time
import os

# Create directory for results if it doesn't exist
if not os.path.exists("results"):
    os.makedirs("results")

# Initialize the TSP problem
print("Creating TSP instance...")
tsp = TSP()
print(f"Created {tsp}")

# Solve with Discrete ACO
print("\nSolving with Discrete ACO...")
discrete_aco = DiscreteACO(tsp)
discrete_path, discrete_distance = discrete_aco.solve()
discrete_aco.plot_solution("results/discrete_aco_solution.png")
discrete_aco.plot_convergence("results/discrete_aco_convergence.png")

# Solve with Distributed ACO
print("\nSolving with Distributed ACO...")
distributed_aco = DistributedACO(tsp)
distributed_path, distributed_distance = distributed_aco.solve()
distributed_aco.plot_solution("results/distributed_aco_solution.png")
distributed_aco.plot_convergence("results/distributed_aco_convergence.png")

# Compare the results
print("\nResults Comparison:")
print(f"Discrete ACO: {discrete_distance:.2f} (time: {discrete_aco.execution_time:.2f}s)")
print(f"Distributed ACO: {distributed_distance:.2f} (time: {distributed_aco.execution_time:.2f}s)")
print(f"Improvement with Distributed ACO: {(discrete_distance - distributed_distance) / discrete_distance * 100:.2f}%")

# Create comparison table as text file instead of CSV to avoid pandas dependency
with open("results/comparison_table.txt", "w") as f:
    f.write("Algorithm,Best Distance,Execution Time (s),% Improvement vs Discrete\n")
    f.write(f"Discrete ACO,{discrete_distance:.2f},{discrete_aco.execution_time:.2f},0.00\n")
    improvement = (discrete_distance - distributed_distance) / discrete_distance * 100
    f.write(f"Distributed ACO,{distributed_distance:.2f},{distributed_aco.execution_time:.2f},{improvement:.2f}\n")

# Plot convergence comparison
plt.figure(figsize=(12, 8))
plt.plot(range(1, len(discrete_aco.history) + 1), discrete_aco.history, 'b-', label='Discrete ACO')
plt.plot(range(1, len(distributed_aco.history) + 1), distributed_aco.history, 'r-', label='Distributed ACO')
plt.title(f"Convergence Comparison\nDiscrete: {discrete_distance:.2f}, Distributed: {distributed_distance:.2f}")
plt.xlabel("Iteration")
plt.ylabel("Best Distance")
plt.legend()
plt.grid(True)
plt.savefig("results/convergence_comparison.png")
plt.close()

# Plot final solutions side by side
plt.figure(figsize=(12, 6))

# Plot Discrete ACO solution
plt.subplot(1, 2, 1)
x = [tsp.cities[i].x for i in discrete_path]
y = [tsp.cities[i].y for i in discrete_path]
x.append(tsp.cities[discrete_path[0]].x)
y.append(tsp.cities[discrete_path[0]].y)
plt.plot(x, y, 'bo-', markersize=5)
plt.title(f"Discrete ACO (Distance: {discrete_distance:.2f})")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)

# Plot Distributed ACO solution
plt.subplot(1, 2, 2)
x = [tsp.cities[i].x for i in distributed_path]
y = [tsp.cities[i].y for i in distributed_path]
x.append(tsp.cities[distributed_path[0]].x)
y.append(tsp.cities[distributed_path[0]].y)
plt.plot(x, y, 'ro-', markersize=5)
plt.title(f"Distributed ACO (Distance: {distributed_distance:.2f})")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)

plt.tight_layout()
plt.savefig("results/solutions_comparison.png")
plt.close()

print("\nExperiment completed! Results saved to 'results' directory.")