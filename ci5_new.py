import numpy as np

# Cities (x,y)
cities = np.random.randint(0, 100, (8, 2))

# Distance matrix
n = len(cities)
dist = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        dist[i][j] = np.linalg.norm(cities[i] - cities[j])

# ACO parameters
ants = 10
iters = 50
alpha = 1      # pheromone importance
beta = 2       # distance importance
evap = 0.5

pheromone = np.ones((n, n))

best_route = None
best_cost = 999999

for _ in range(iters):
    routes = []
    costs = []

    # Each ant builds a route
    for _ in range(ants):
        route = [0]
        while len(route) < n:
            current = route[-1]
            unvisited = [i for i in range(n) if i not in route]
            probs = []

            for city in unvisited:
                p = (pheromone[current][city] ** alpha) * \
                    ((1 / dist[current][city]) ** beta)

                probs.append(p)

            probs = probs / np.sum(probs)

            next_city = np.random.choice(unvisited, p=probs)

            route.append(next_city)

        # Total distance
        cost = 0

        for i in range(n - 1):
            cost += dist[route[i]][route[i + 1]]

        cost += dist[route[-1]][route[0]]

        routes.append(route)
        costs.append(cost)

        # Best solution
        if cost < best_cost:
            best_cost = cost
            best_route = route

    # Evaporation
    pheromone *= (1 - evap)

    # Pheromone update
    for route, cost in zip(routes, costs):
        for i in range(n - 1):
            a, b = route[i], route[i + 1]
            pheromone[a][b] += 1 / cost
            pheromone[b][a] += 1 / cost


print("Cities:\n", cities)

print("\nBest Route:")
print(best_route, "->", best_route[0])

print("\nBest Distance:", round(best_cost, 2))
