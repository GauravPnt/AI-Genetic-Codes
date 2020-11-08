import numpy as np
from math import *
import matplotlib.pyplot as plt
DEBUG = False

rng = np.random.default_rng()


def single_point_crossover(x, y):
    if DEBUG:
        print("Single point crossover: ", x, y)
    rnd = rng.choice(high_bit + 1)
    flip = (((1 << (rnd + 1)) - 1) & x) ^ (((1 << (rnd + 1)) - 1) & y)
    if DEBUG:
        print('Index: ', rnd)
        print([x ^ flip, y ^ flip])
    return [x ^ flip, y ^ flip]


def two_point_crossover(x, y):
    if DEBUG:
        print("Two point crossover: ", x, y)
    rnd = rng.choice(high_bit + 1, size=2, replace=False)
    mask = ((1 << (rnd[1] + 1)) - 1) ^ ((1 << (rnd[0] + 1)) - 1)
    flip = (mask & x) ^ (mask & y)
    if DEBUG:
        print('Index: ', rnd)
        print([x ^ flip, y ^ flip])
    return [x ^ flip, y ^ flip]


def multi_point_crossover(x, y):
    if DEBUG:
        print("Multi point crossover: ", x, y)
    rnd = rng.choice(high_bit + 1, size=3, replace=False)
    mask = (1 << (high_bit + 1)) - 1
    for j in rnd:
        mask ^= ((1 << (j + 1)) - 1)
    flip = (mask & x) ^ (mask & y)
    if DEBUG:
        print('Index: ', rnd)
        print([x ^ flip, y ^ flip])
    return [x ^ flip, y ^ flip]


def uniform_crossover(x, y):
    if DEBUG:
        print("Uniform crossover: ", x, y)
    rnd = rng.choice(2, size=high_bit + 1)
    for i in range(len(rnd)):
        if rnd[i] > 0:
            flip = (((1 << (i + 1)) - 1) & x) ^ (((1 << (i + 1)) - 1) & y)
            x ^= flip
            y ^= flip
    if DEBUG:
        print('Index: ', rnd)
        print([x, y])
    return [x, y]


crossover = [single_point_crossover, two_point_crossover, multi_point_crossover, uniform_crossover]


def bit_flip_mutation(x):
    if DEBUG:
        print('Bit Flip Mutation: ', x)
    rnd = np.random.rand(high_bit + 1)
    for idx in range(0, len(rnd)):
        if rnd[idx] <= 1 / high_bit:
            x ^= (1 << idx)
    if DEBUG:
        print('Index: ', rnd)
        print(x)
    return x


def swap_mutation(x):
    if DEBUG:
        print('Swap Mutation: ', x)
    rnd = rng.choice(range(1, high_bit + 1), size=2, replace=False)
    mask1 = (((1 << rnd[0]) & x) >= 1)
    mask2 = (((1 << rnd[1]) & x) >= 1)
    x = (x ^ (x & (1 << rnd[1]))) | (mask1 << rnd[1])
    x = (x ^ (x & (1 << rnd[0]))) | (mask2 << rnd[0])
    if DEBUG:
        print('Index: ', rnd)
        print(x)
    return x


def scramble_mutation(x):
    if DEBUG:
        print('Scramble Mutation: ', x)
    bits = []
    rnd = rng.choice(high_bit + 1, size=2, replace=False)
    rnd.sort()
    for i in range(rnd[0], rnd[1] + 1):
        bits.append(x & (1 << i) == 1)
    bits = np.array(bits)
    np.random.shuffle(bits)
    for i in range(rnd[0], rnd[1] + 1):
        x = (x ^ (x & (1 << i))) | (bits[i - rnd[0]] << i)
    if DEBUG:
        print('Index: ', rnd)
        print(x)
    return x


mutation = [bit_flip_mutation, swap_mutation, scramble_mutation]


class Genetic:
    def __init__(self, fitness_fn, low, high):
        self.fitness_fn = fitness_fn
        self.population = rng.choice(range(low, high + 1), size=5, replace=False)
        self.iter = 0
        self.gen = []
        self.fittest = []
        self.fittest_x = None
        self.global_fittest_x = None
        self.global_fittest = None
        self.parents = None
        self.visit = set()
        self.generate_population()

    def eval_and_generate_population(self):
        fitness = []
        for x in self.population:
            fx = eval(self.fitness_fn)
            fitness.append(fx)
        fitness = np.array(fitness)
        max_fittest = max(fitness)
        total = sum(fitness)
        self.gen.append(self.iter)
        self.fittest.append(max_fittest)
        for i in range(0, len(self.population)):
            if fitness[i] == max_fittest:
                self.fittest_x = self.population[i]
                break
        if self.global_fittest is None:
            self.global_fittest = max_fittest
            self.global_fittest_x = self.fittest_x
        elif self.global_fittest < max_fittest:
            self.global_fittest = max_fittest
            self.global_fittest_x = self.fittest_x
        prob = fitness / total
        p1 = rng.choice(a=self.population, size=3, p=prob, replace=False)
        self.parents = p1
        new_population = []
        visit = set()
        flag = False
        for i in p1:
            for j in p1:
                if i == j:
                    continue
                for fnc in crossover:
                    genes = fnc(i, j)
                    for pivot in genes:
                        x = pivot
                        f = eval(self.fitness_fn)
                        max_x = None
                        for fnm in mutation:
                            if max_x is None:
                                max_x = pivot
                            x = fnm(pivot)
                            fm = eval(self.fitness_fn)
                            if fm > f:
                                f = fm
                                max_x = x
                        if max_x not in self.visit:
                            if flag is False:
                                flag = True
                            self.visit.add(max_x)
                        if max_x not in visit:
                            new_population.append(max_x)
                            visit.add(max_x)
        self.population = new_population
        self.iter += 1
        return flag

    def generate_population(self):
        print('\nGeneration {} :'.format(self.iter), self.population)
        while self.eval_and_generate_population():
            print('Fittest X :', self.fittest_x)
            print('\nParents :', self.parents)
            print('Generation {} :\n'.format(self.iter), self.population)
            continue
        print('Fittest X :', self.fittest_x)


Fitness_fn = input()
Low = int(input())
High = int(input())
high_bit = int(log2(High))

g = Genetic(Fitness_fn, Low, High)
print("\nAns: ", g.global_fittest_x)
print("Fitness: ", g.global_fittest)
plt.plot(g.gen, g.fittest)
plt.show()
