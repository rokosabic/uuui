import argparse
import numpy as np

def load_data(path):
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    X = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)
    return X, y

# sigmoid funkc u skrivenim slojevima
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# kv odstupanje izmedu stvarnih i predvidenih vr
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

class NeuralNetwork:
    # init neuronske mreze (arhitektura, oblik slojeva, broj paramsa, tezine i bias)
    def __init__(self, input_dim, architecture):
        self.architecture = architecture
        self.shapes = self._get_shapes(input_dim, architecture)
        self.n_params = sum((np.prod(shape) for shape in self.shapes['weights'])) + sum(self.shapes['biases'])
        self.weights, self.biases = self._init_params()

    # oblikuje slojeve
    def _get_shapes(self, input_dim, architecture):
        if architecture == '5s':
            layers = [input_dim, 5, 1]
        elif architecture == '20s':
            layers = [input_dim, 20, 1]
        elif architecture == '5s5s':
            layers = [input_dim, 5, 5, 1]
        else:
            raise ValueError('Nepodržana arhitektura mreže')

        weights = [(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        biases = [layers[i+1] for i in range(len(layers)-1)]

        return {'weights': weights, 'biases': biases}

    # init tezina (0, 0.01) i biasa (sve 0)
    def _init_params(self):
        weights = [np.random.normal(0, 0.01, size=shape) for shape in self.shapes['weights']]
        biases = [np.zeros((1, bsize)) for bsize in self.shapes['biases']]

        return weights, biases

    # vraca 1 vektor svih tezina i biasa
    # enkodira mrezu u jedinku gen alg
    def get_params_vector(self):
        return np.concatenate([w.flatten() for w in self.weights] + [b.flatten() for b in self.biases])

    # postavlja tezine i biase 1 vektora
    # dekodiranje jedinke iz gen alg u mrezu
    def set_params_vector(self, vector):
        weights = []
        biases = []
        idx = 0

        for shape in self.shapes['weights']:
            size = np.prod(shape)
            weights.append(vector[idx:idx+size].reshape(shape))
            idx += size

        for bsize in self.shapes['biases']:
            biases.append(vector[idx:idx+bsize].reshape(1, bsize))
            idx += bsize

        self.weights = weights
        self.biases = biases

    # racuna izlaz mreze (skriveni sloj = sigm + lin, izlazni = lin (bez akt))
    def forward(self, X):
        a = X

        for i in range(len(self.weights) - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = sigmoid(z)

        return np.dot(a, self.weights[-1]) + self.biases[-1]

class GeneticAlgorithm:
    # init populacije jedinki / hiperparametri / podaci
    def __init__(self, nn_template, popsize, elitism, p, K, n_iter, X_train, y_train):
        self.nn_template = nn_template
        self.popsize = popsize
        self.elitism = elitism
        self.p = p
        self.K = K
        self.n_iter = n_iter
        self.X_train = X_train
        self.y_train = y_train
        self.n_params = nn_template.n_params

        self.population = [self._random_individual() for _ in range(popsize)]

    # sluc jedinka
    def _random_individual(self):
        return np.random.normal(0, 0.01, size=self.n_params)

    # racuna fitness jedinke, postavi tezine mreze na vrijednosti jedinke, neg. mse
    def _fitness(self, individual):
        nn = self._make_nn(individual)
        y_pred = nn.forward(self.X_train)
        mse = mean_squared_error(self.y_train, y_pred)

        return -mse

    # kreira novu mrezu i postavlja joj tezine iz params
    def _make_nn(self, params):
        nn = NeuralNetwork(self.nn_template.shapes['weights'][0][0], self.nn_template.architecture)
        nn.set_params_vector(params)

        return nn

    # bira sluc index prema vjerojatnostima koje su proporcionalne fitnessu
    def _select(self, fitnesses):
        fitnesses = np.array(fitnesses)
        min_fit = np.min(fitnesses)

        if min_fit < 0:
            fitnesses = fitnesses - min_fit + 1e-10

        probs = fitnesses / np.sum(fitnesses)
        idx = np.random.choice(self.popsize, p=probs)

        return idx

    # stvara dijete (arit. sr.)
    def _crossover(self, parent1, parent2):
        return (parent1 + parent2) / 2

    # mutira tezine s P, dodaje Gaussov sum
    def _mutate(self, individual):
        mask = np.random.rand(self.n_params) < self.p
        noise = np.random.normal(0, self.K, size=self.n_params)
        individual = individual.copy()
        individual[mask] += noise[mask]

        return individual

    # glavni evol. ciklus
    # eval fitness
    # prenosi najb jedinke
    # generira novu populaciju krizanjem i mutacijom
    # svakih 2000 gen ispisuje gresku najb jedinke na tr skupu
    # vraca najbolju jedinku (tezine)
    def run(self):
        best_train_errors = []

        for gen in range(1, self.n_iter+1):
            fitnesses = [self._fitness(ind) for ind in self.population]

            elite_idx = np.argsort(fitnesses)[-self.elitism:]
            new_population = [self.population[i].copy() for i in elite_idx]

            while len(new_population) < self.popsize:
                i1 = self._select(fitnesses)
                i2 = self._select(fitnesses)
                child = self._crossover(self.population[i1], self.population[i2])
                child = self._mutate(child)
                new_population.append(child)
            self.population = new_population

            if gen % 2000 == 0 or gen == self.n_iter:
                best_idx = np.argmax(fitnesses)
                nn = self._make_nn(self.population[best_idx])
                y_pred = nn.forward(self.X_train)
                train_error = mean_squared_error(self.y_train, y_pred)
                print(f"[Train error @{gen}]: {train_error:.6f}")
                best_train_errors.append(train_error)

        fitnesses = [self._fitness(ind) for ind in self.population]
        best_idx = np.argmax(fitnesses)

        return self.population[best_idx]

# validna arhitektura?
def parse_architecture(arch_str, input_dim):
    if arch_str not in ['5s', '20s', '5s5s']:
        raise ValueError('Nepodržana arhitektura mreže')

    return arch_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--nn', required=True)
    parser.add_argument('--popsize', type=int, required=True)
    parser.add_argument('--elitism', type=int, required=True)
    parser.add_argument('--p', type=float, required=True)
    parser.add_argument('--K', type=float, required=True)
    parser.add_argument('--iter', type=int, required=True)
    args = parser.parse_args()

    X_train, y_train = load_data(args.train)
    X_test, y_test = load_data(args.test)
    input_dim = X_train.shape[1]
    arch = parse_architecture(args.nn, input_dim)
    nn_template = NeuralNetwork(input_dim, arch)

    ga = GeneticAlgorithm(
        nn_template=nn_template,
        popsize=args.popsize,
        elitism=args.elitism,
        p=args.p,
        K=args.K,
        n_iter=args.iter,
        X_train=X_train,
        y_train=y_train
    )
    best_params = ga.run()

    nn = NeuralNetwork(input_dim, arch)
    nn.set_params_vector(best_params)
    y_pred = nn.forward(X_test)
    test_error = mean_squared_error(y_test, y_pred)
    print(f"[Test error]: {test_error:.6f}")

if __name__ == '__main__':
    main()
