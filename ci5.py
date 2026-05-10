import numpy as np
import random

def generate_cities(n=8):
    """Generate random city coordinates."""
    random.seed(55)
    return [(random.randint(0, 100), random.randint(0, 100)) for _ in range(n)]

def distance(c1, c2):
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

def build_distance_matrix(cities):
    n = len(cities)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i][j] = distance(cities[i], cities[j])
    return dist

def ant_colony_optimization(cities, n_ants=20, n_iter=100, alpha=1.0, beta=3.0, evap=0.5, Q=100):
    n = len(cities)
    dist = build_distance_matrix(cities)
    pheromone = np.ones((n, n))
    best_route, best_length = None, float('inf')

    for iteration in range(n_iter):
        all_routes, all_lengths = [], []

        for _ in range(n_ants):
            # Build route
            visited = [0]
            while len(visited) < n:
                curr = visited[-1]
                unvisited = [c for c in range(n) if c not in visited]
                # Calculate probabilities
                probs = []
                for j in unvisited:
                    tau = pheromone[curr][j] ** alpha
                    eta = (1.0 / dist[curr][j]) ** beta if dist[curr][j] > 0 else 1e10
                    probs.append(tau * eta)
                probs = np.array(probs) / sum(probs)
                next_city = np.random.choice(unvisited, p=probs)
                visited.append(next_city)

            # Calculate tour length
            length = sum(dist[visited[i]][visited[i+1]] for i in range(n-1)) + dist[visited[-1]][visited[0]]
            all_routes.append(visited)
            all_lengths.append(length)

            if length < best_length:
                best_length = length
                best_route = visited[:]

        # Update pheromones
        pheromone *= (1 - evap)  # Evaporation
        for route, length in zip(all_routes, all_lengths):
            deposit = Q / length
            for i in range(n - 1):
                pheromone[route[i]][route[i+1]] += deposit
                pheromone[route[i+1]][route[i]] += deposit
            pheromone[route[-1]][route[0]] += deposit
            pheromone[route[0]][route[-1]] += deposit

        if iteration % 20 == 0:
            print(f"Iteration {iteration:3d} | Best length: {best_length:.2f}")

    return best_route, best_length

# Run
print("=== Ant Colony Optimization - Traveling Salesman Problem ===\n")
cities = generate_cities(8)
print("Cities (x, y):")
for i, c in enumerate(cities):
    print(f"  City {i}: {c}")
print()

best_route, best_length = ant_colony_optimization(cities)

print(f"\nBest Route: {' -> '.join(map(str, best_route))} -> {best_route[0]}")
print(f"Total Distance: {best_length:.2f}")
