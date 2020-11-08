import random
import numpy as np
import matplotlib.pyplot as plt


class SBX:
    def __init__(self, iter, low, high, fn, eta, pm, pc):
        self.epochs = iter
        self.low = low
        self.high = high
        self.fn = fn
        self.answer = None
        self.eta = eta
        self.pm = pm
        self.pc = pc
        self.picked = []

    def solve(self):
        fitness = []
        population = []
        for i in range(6):
            population.append(random.sample(range(low, high), 4))

        for it in range(self.epochs):
            print("Generation", it, '\n', np.array(population))
            print()
            for _ in range(100):
                print("=", end="")
            print('\n')
            for i in range(len(population)):
                w = population[i][0]
                x = population[i][1]
                y = population[i][2]
                z = population[i][3]
                fitness.append(eval(self.fn))
            minfitness = min(fitness)
            self.picked.append(minfitness)
            if self.answer is None or self.answer > minfitness:
                self.answer = minfitness
            generation = []
            population = np.array(population)

            # tournament selection
            for i in range(6):
                idx1 = int(np.random.random()*1000) % 6
                idx2 = int(np.random.random()*1000) % 6
                while idx1 == idx2:
                    idx1 = int(np.random.random()*1000) % 6
                    idx2 = int(np.random.random()*1000) % 6

                w = population[idx1][0]
                x = population[idx1][1]
                y = population[idx1][2]
                z = population[idx1][3]
                parent1value = eval(self.fn)

                w = population[idx2][0]
                x = population[idx2][1]
                y = population[idx2][2]
                z = population[idx2][3]
                parent2value = eval(self.fn)

                if parent1value <= parent2value:
                    generation.append(population[idx1])
                else:
                    generation.append(population[idx2])

            generation = np.array(generation)
            generationbeforemutation = []

            # crossover
            for i in range(int(len(generation)/2)):
                idx1 = int(np.random.random()*1000) % 6
                idx2 = int(np.random.random()*1000) % 6
                while idx1 == idx2:
                    idx1 = int(np.random.random()*1000) % 6
                    idx2 = int(np.random.random()*1000) % 6

                if np.random.random() > self.pc:
                    continue

                u = [np.random.random() for j in range(4)]
                beta = []
                eta = self.eta+1
                for ui in u:
                    if ui <= 0.5:
                        beta.append((2*ui)**(1/eta))
                    else:
                        beta.append(1/((2*(1-ui))**(1/eta)))
                beta = np.array(beta)
                x1 = (0.5*((1+beta)*np.array(generation[idx1])+(1-beta)*np.array(generation[idx2])))
                x2 = (0.5*((1-beta)*np.array(generation[idx1])+(1+beta)*np.array(generation[idx2])))

                generationbeforemutation.append(x1)
                generationbeforemutation.append(x2)

            # mutation
            generationaftermutation = list(population)
            for i in range(len(generationbeforemutation)):
                newval = generationbeforemutation[i]
                for j in range(4):
                    if np.random.random() <= self.pm:
                        delta = None
                        r = np.random.random()
                        eta = self.eta+1
                        if r < 0.5:
                            delta = ((2*r)**(1/eta))-1
                        else:
                            delta = 1-((2*(1-r))**(1/eta))
                        newval[j] = generationbeforemutation[i][j] + (self.high-self.low)*delta
                    newval[j] = max(self.low, min(newval[j], self.high))
                generationaftermutation.append(newval)
            generationaftermutation = sorted(generationaftermutation, key=lambda x: x[0]+x[1]+x[2]+x[3])[:6]
            population = generationaftermutation


fn = "w*w+x*x+y*y+z*z"
print("x1^2 + x2^2 + x3^2 + x4^2")
low = int(input())
high = int(input())
g = SBX(60, low, high, fn, 15, 0.2, 0.8)
g.solve()
print("Answer = ", g.answer)
plt.plot(g.picked)
plt.xlabel("Generation")
plt.ylabel("Function = w*w+x*x+y*y+z*z")
plt.show()
