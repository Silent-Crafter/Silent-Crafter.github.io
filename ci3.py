import numpy as np

def objective(x):
    return x ** 2

def clonal_selection(pop_size=20, n_clones=5, n_gen=50, mut_rate=0.5, x_range=(-10, 10)):
    population = np.random.uniform(*x_range, size=pop_size)

    for gen in range(n_gen):
        fitness = objective(population)

        # Select top antibodies
        sorted_idx = np.argsort(fitness)[:n_clones]
        selected = population[sorted_idx]

        clones = []
        for i, ab in enumerate(selected):
            n = n_clones - i
            for _ in range(n):
                mutation = np.random.normal(0, mut_rate / (i + 1))
                clone = np.clip(ab + mutation, *x_range)
                clones.append(clone)

        clones = np.array(clones)
        clone_fitness = np.array([objective(c) for c in clones])

        all_individuals = np.concatenate([clones, np.random.uniform(*x_range, pop_size // 4)])
        all_fitness = np.array([objective(x) for x in all_individuals])
        best_idx = np.argsort(all_fitness)[:pop_size]
        population = all_individuals[best_idx]

        best = population[0]
        if gen % 10 == 0:
            print(f"Gen {gen:3d} | Best x = {best:.6f} | f(x) = {objective(best):.6f}")

    return population[0]

print("=== Clonal Selection Algorithm ===")
print("Minimizing f(x) = x^2\n")
result = clonal_selection()
print(f"\nOptimal solution: x = {result:.6f}, f(x) = {objective(result):.6f}")
