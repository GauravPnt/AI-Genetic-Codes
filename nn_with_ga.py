import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_iris

class Neural:

  def __init__(self):
    self.sigmoid = torch.nn.Sigmoid()
    self.loss = torch.nn.CrossEntropyLoss()
    self.w0_shape = (4, 4)
    self.w1_shape = (4, 3)

  def forward(self, ip, w0, w1):
    x = torch.matmul(ip, w0)
    x = self.sigmoid(x)
    x = torch.matmul(x, w1)
    x = self.sigmoid(x)
    return x

  def eval(self, chromosome, ip, label):
    w0 = torch.tensor(chromosome[:self.w0_shape[0] * self.w0_shape[1]]).view(self.w0_shape[0], self.w0_shape[1])
    w1 = torch.tensor(chromosome[self.w0_shape[0] * self.w0_shape[1]:]).view(self.w1_shape[0], self.w1_shape[1])
    x = self.forward(ip, w0, w1)
    return self.loss(x, label).item()

class SBX:
  def __init__(self, iter, low, high, fn, eta, pm, pc, p_size):
    self.p_size = p_size
    self.w0_shape = (4, 4)
    self.w1_shape = (4, 3)
    self.chromosome = None
    self.neural = Neural()
    self.data = load_iris()
    self.features = torch.tensor(data.data)
    self.label = torch.tensor(data.target)
    self.data_size = self.data.data.shape[0]
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
    population = []
    for i in range(self.p_size):
        population.append(np.random.randn(self.w0_shape[1] * self.w0_shape[0] + self.w1_shape[1] * self.w1_shape[0]))

    for it in range(self.epochs):
      fitness = []
      print("Generation", it, '\n', np.array(population))
      print()
      for _ in range(100):
        print("=", end="")
      print('\n')
      for i in range(self.p_size):
        fitness.append(self.neural.eval(population[i], self.features, self.label))
      minfitness = min(fitness)
      idx = np.argmin(fitness)
      self.picked.append(minfitness)
      if self.answer is None or self.answer > minfitness:
        self.answer = minfitness
        self.chromosome = population[idx]
      generation = []
      population = np.array(population)

      # tournament selection
      for i in range(self.p_size):
        idx1 = int(np.random.random()*1000) % self.p_size
        idx2 = int(np.random.random()*1000) % self.p_size
        while idx1 == idx2:
          idx1 = int(np.random.random()*1000) % self.p_size
          idx2 = int(np.random.random()*1000) % self.p_size

        parent1value = self.neural.eval(population[idx1], self.features, self.label)
        parent2value = self.neural.eval(population[idx2], self.features, self.label)

        if parent1value <= parent2value:
          generation.append(population[idx1])
        else:
          generation.append(population[idx2])

      generation = np.array(generation)
      generationbeforemutation = []

      # crossover
      for i in range(int(len(generation)/2)):
        idx1 = int(np.random.random()*1000) % self.p_size
        idx2 = int(np.random.random()*1000) % self.p_size
        while idx1 == idx2:
          idx1 = int(np.random.random()*1000) % self.p_size
          idx2 = int(np.random.random()*1000) % self.p_size

        if np.random.random() > self.pc:
          continue

        u = [np.random.random() for j in range(self.w0_shape[1] * self.w0_shape[0] + self.w1_shape[1] * self.w1_shape[0])]
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
        for j in range(self.w0_shape[1] * self.w0_shape[0] + self.w1_shape[1] * self.w1_shape[0]):
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
      generationaftermutation = sorted(generationaftermutation, key=lambda x: self.neural.eval(x, self.features, self.label))[:self.p_size]
      population = generationaftermutation

    w0 = torch.tensor(self.chromosome[:self.w0_shape[0] * self.w0_shape[1]]).view(self.w0_shape[0], self.w0_shape[1])
    w1 = torch.tensor(self.chromosome[self.w0_shape[0] * self.w0_shape[1]:]).view(self.w1_shape[0], self.w1_shape[1])
    y = self.neural.forward(self.features, w0, w1)
    accuracy = torch.sum(torch.eq(torch.max(y, 1).indices, self.label)).item() / self.data_size
    print("Accuracy: ", accuracy)


fn = "w*w+x*x+y*y+z*z"
print("x1^2 + x2^2 + x3^2 + x4^2")
low = int(input())
high = int(input())
g = SBX(60, low, high, fn, 15, 0.2, 0.8, 6)
g.solve()
print("Answer = ", g.answer)
plt.plot(g.picked)
plt.xlabel("Generation")
plt.ylabel("Loss")
plt.show()
