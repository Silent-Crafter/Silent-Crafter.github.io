from deap import base, creator, tools, algorithms
import random
import numpy as np

# Problem setup
IND_SIZE = 50  # Length of binary string
POP_SIZE = 100
N_GEN = 30
CX_PROB = 0.7  # Crossover probability
MUT_PROB = 0.2  # Mutation probability

# Define fitness and individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Toolbox setup
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Fitness function: count number of 1s
def eval_onemax(individual):
    return (sum(individual),)

toolbox.register("evaluate", eval_onemax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Statistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("max", np.max)
stats.register("min", np.min)

# Run evolution
print("=== DEAP: Evolutionary Algorithm (OneMax Problem) ===")
print(f"Goal: Maximize ones in a binary string of length {IND_SIZE}\n")

random.seed(42)
population = toolbox.population(n=POP_SIZE)
hall_of_fame = tools.HallOfFame(1)

result_pop, logbook = algorithms.eaSimple(
    population, toolbox,
    cxpb=CX_PROB, mutpb=MUT_PROB, ngen=N_GEN,
    stats=stats, halloffame=hall_of_fame, verbose=True
)

# Results
best = hall_of_fame[0]
print(f"\nBest Individual: {''.join(map(str, best))}")
print(f"Fitness (number of 1s): {best.fitness.values[0]}/{IND_SIZE}")
