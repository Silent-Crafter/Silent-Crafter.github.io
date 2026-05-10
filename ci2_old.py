import math
import numpy as np
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

from typing import Any

SEED = 55
random.seed(SEED)

def generate_individual():
    return {
        'hl': random.randint(1, 5),         # no. of hidden neurons
        'lr': random.uniform(0.001, 0.1),   # learning rate
        'm':  random.uniform(0.0, 0.9),     # momentum
    }

def fitness(individual: dict[str, Any], x_train, x_test, y_train, y_test):
    model = MLPClassifier(
        hidden_layer_sizes=(individual['hl'],),
        learning_rate_init=individual['lr'],
        momentum=individual['m'],
        max_iter=500,
        random_state=SEED
    )

    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    acc = accuracy_score(y_test, preds)
    return acc

def selection(population, fitness_scores):
    s = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
    return s[:len(s)//2]

def crossover(parent1, parent2):
    return {
        'hl': parent1['hl'],
        'lr': parent2['lr'],
        'm': parent1['m']
    }

def mutation(individual):
    # only changing the learning rate
    if random.random() < 0.1:
        individual['lr'] = random.uniform(0.001, 0.1)

    return individual


if __name__ == "__main__":
    np.random.seed(SEED)

    scaler = MinMaxScaler()
    iris = load_iris()

    X = iris.data
    Y = iris.target

    X = scaler.fit_transform(X)

    split_data = train_test_split(X, Y, test_size=0.2, random_state=SEED)

    population_size = 10
    generations = 10

    population = [generate_individual() for _ in range(population_size)]

    fitness_history = []

    best_overall = None
    best_overall_fitness = -math.inf
    best_generation = 0

    for gen in range(generations):
        fitness_scores = [fitness(ind, *split_data) for ind in population]

        gen_best_fitness = max(fitness_scores)
        gen_best_ind = population[fitness_scores.index(gen_best_fitness)]

        fitness_history.append(gen_best_fitness)

        print()
        print('-'*50)
        print("Generation", gen+1)
        print("\tBest Fitness: {:.2f}".format(gen_best_fitness))
        print("\tBest individual:", gen_best_ind)
        print('-'*50)

        if gen_best_fitness >= best_overall_fitness:
            best_overall_fitness = gen_best_fitness
            best_overall = gen_best_ind
            best_generation = gen + 1

        selected = selection(population, fitness_scores)
        
        # Create next generation
        children = []
        while len(children) < population_size:
            p1, p2 = random.sample(selected, 2)
            child = crossover(p1[0], p2[0])
            child = mutation(child)
            children.append(child)
        
        population = children

    print("\n===== FINAL RESULT =====")
    print("Best Generation:", best_generation)
    print("Best Fitness: {:.2f}".format(best_overall_fitness))
    print("Best Parameters:", best_overall)
