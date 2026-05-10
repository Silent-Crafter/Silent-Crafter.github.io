import pygad

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data = load_iris()

X = data.data
y = data.target

scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def fitness_function(ga_instance, solution, solution_idx):
    model = MLPClassifier(hidden_layer_sizes=(int(solution[0]),), alpha=solution[1])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

gene_space = [
    range(2, 100),                  # no. of neurons
    {'low': 0.0001, 'high': 0.1}    # learning rate
]

ga_instance = pygad.GA(
        num_generations=3, 
        num_parents_mating=2,
        sol_per_pop=4,
        num_genes=2,
        gene_space=gene_space,
        mutation_type=None,
        fitness_func=fitness_function
)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()

print(f'\n----- BEST PARAMETERS FOUND ------')
print(f'Optimal neurons: {solution[0]}')
print(f'Optimal lr: {solution[1]}')
print(f'Best accuracy: {solution_fitness}')

