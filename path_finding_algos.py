import pygame
from pygame.locals import *
import queue
import copy
import math
import time
import math
import random

WIDTH = 600
ROWS = 30
CELL_WIDTH = WIDTH // ROWS

dx = [1, -1, 0]
inf = 10000000000000000


class Coord:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)
        self.parent = None
        self.heuristic = inf
        self.cost = inf

    def __lt__(self, other):
        return True


def convert(x):
    return int(x) * CELL_WIDTH + CELL_WIDTH // 2


def check(x, y):
    return 0 <= convert(y) < WIDTH and 0 <= convert(x) < WIDTH


class App:

    def __init__(self, s, e):
        self.screen = pygame.display.set_mode((WIDTH, WIDTH))
        self.running = True
        self.start = s
        self.final = e
        self.wall = set()
        self.generate_walls()
        self.run()

    def generate_walls(self):
        for i in range(3 * ROWS // 4):
            if i != 0 and i != ROWS - 1:
                self.wall.add((14, i))
            for j in range(ROWS):
                if random.random() < 0.05:
                    if (i == self.start.x and j == self.start.y) or (i == self.final.x and j == self.final.y):
                        continue
                    self.wall.add((i, j))

    def eval_heuristic(self, curr, over=False):
        if (curr.x, curr.y) in self.wall:
            return inf
        x, y = curr.x, curr.y
        fx, fy = self.final.x, self.final.y
        if over:
            return (x - fx) * (x - fx) + (y - fy) * (y - fy)
        return math.floor(math.sqrt((x - fx) * (x - fx) / 2 + (y - fy) * (y - fy) / 2))

    def eval_cost(self, curr):
        if (curr.x, curr.y) in self.wall:
            return inf
        return 1

    def fill_screen(self):
        # print("\nCost:")
        for y in range(ROWS):
            for x in range(ROWS):
                if (x, y) in self.wall:
                    # print("  ", end=' ')
                    pygame.draw.circle(self.screen, (0, 0, 0),
                                       (convert(x), convert(y)), CELL_WIDTH // 2)
                    continue
                pygame.draw.circle(self.screen, (0, 0, 255), (convert(x), convert(y)), CELL_WIDTH // 2)
                # print("%2d" % self.eval_cost(Coord(x, y)), end=' ')
            # print()

    def evaluate_best_first(self):
        curr = self.start
        pq = queue.PriorityQueue()
        h = self.eval_heuristic(curr, True)
        curr.heuristic = h
        curr.cost = 0
        pq.put((h, curr))
        visit = set()
        self.best_first_generated = dict()
        f = None
        while not pq.empty():
            curr = pq.get()[1]
            if (curr.x, curr.y) == (self.final.x, self.final.y):
                f = curr
                break
            visit.add((curr.x, curr.y))
            for i in dx:
                for j in dx:
                    if i == j == 0:
                        continue
                    if not check(curr.x + i, curr.y + j):
                        continue
                    temp = None
                    if (curr.x + i, curr.y + j) in self.best_first_generated:
                        if temp is None:
                            temp = self.best_first_generated[(
                                curr.x + i, curr.y + j)]
                    else:
                        if temp is None:
                            temp = Coord(curr.x + i, curr.y + j)
                            self.best_first_generated[(temp.x, temp.y)] = temp
                    if (temp.x, temp.y) in visit:
                        continue
                    sh = self.eval_heuristic(temp, True)
                    if sh < temp.heuristic:
                        temp.heuristic = sh
                        temp.parent = curr
                        temp.cost = curr.cost + self.eval_cost(temp)
                        pq.put((sh, temp))
        while f is not None:
            self.best_first_path.append(f)
            f = f.parent
        self.best_first_path.reverse()
        self.best_first_time = time.time() - self.best_first_time

    def evaluate_dijkstra(self):
        curr = self.start
        pq = queue.PriorityQueue()
        curr.cost = 0
        curr.heuristic = self.eval_heuristic(curr)
        pq.put((0, curr))
        visit = set()
        self.dijkstra_generated = dict()
        self.dijkstra_generated[(curr.x, curr.y)] = curr
        f = None
        while not pq.empty():
            curr = pq.get()[1]
            # input()
            if (curr.x, curr.y) == (self.final.x, self.final.y):
                f = curr
            self.dijkstra_generated[(curr.x, curr.y)] = curr
            for i in dx:
                for j in dx:
                    if i == j == 0:
                        continue
                    if not check(curr.x + i, curr.y + j):
                        continue
                    temp = None
                    if (curr.x + i, curr.y + j) in self.dijkstra_generated:
                        if temp is None:
                            temp = self.dijkstra_generated[(
                                curr.x + i, curr.y + j)]
                    else:
                        if temp is None:
                            temp = Coord(curr.x + i, curr.y + j)
                            self.dijkstra_generated[(temp.x, temp.y)] = temp
                    if (temp.x, temp.y) in visit:
                        continue
                    sc = self.eval_cost(temp) + curr.cost
                    assert self.eval_cost(temp) > 0
                    if sc < temp.cost:
                        temp.heuristic = self.eval_heuristic(temp)
                        temp.parent = curr
                        temp.cost = sc
                        pq.put((sc, temp))
        while f is not None:
            self.path_dijkstra.append(f)
            f = f.parent
        self.path_dijkstra.reverse()
        self.dijkstra_time = time.time() - self.dijkstra_time

    def evaluate_a_star(self, over=False):
        curr = self.start
        pq = queue.PriorityQueue()
        curr.cost = 0
        curr.heuristic = self.eval_heuristic(curr, over)
        print("h", curr.heuristic)
        print("h", curr.x, curr.y)
        print("h", self.final.x, self.final.y)
        pq.put((curr.heuristic, curr))
        visit = set()
        self.a_star_generated = dict()
        self.a_star_generated[(curr.x, curr.y)] = curr
        f = None
        while not pq.empty():
            curr = pq.get()[1]
            # input()
            if (curr.x, curr.y) == (self.final.x, self.final.y):
                f = curr
                break
            self.a_star_generated[(curr.x, curr.y)] = curr
            for i in dx:
                for j in dx:
                    if i == j == 0:
                        continue
                    if not check(curr.x + i, curr.y + j):
                        continue
                    temp = None
                    if (curr.x + i, curr.y + j) in self.a_star_generated:
                        if temp is None:
                            temp = self.a_star_generated[(
                                curr.x + i, curr.y + j)]
                    else:
                        if temp is None:
                            temp = Coord(curr.x + i, curr.y + j)
                            self.a_star_generated[(temp.x, temp.y)] = temp
                    if (temp.x, temp.y) in visit:
                        continue
                    sc = self.eval_cost(temp) + curr.cost
                    assert self.eval_cost(temp) > 0
                    if sc < temp.cost:
                        temp.heuristic = self.eval_heuristic(temp, over)
                        temp.parent = curr
                        temp.cost = sc
                        pq.put((sc + temp.heuristic, temp))
        while f is not None:
            self.path_a_star.append(f)
            f = f.parent
        self.path_a_star.reverse()
        self.a_star_time = time.time() - self.a_star_time

    def run(self):
        self.reset_board()
        while self.running:
            pygame.display.update()
            self.events()

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                pressed = pygame.key.get_pressed()
                if pressed[pygame.K_d]:
                    self.draw_dijkstra()
                if pressed[pygame.K_b]:
                    self.draw_best_first_search()
                if pressed[pygame.K_a]:
                    self.draw_a_star()
                if pressed[pygame.K_o]:
                    self.draw_a_star(True)

    def reset_board(self):
        self.fill_screen()
        pygame.draw.circle(self.screen, (255, 0, 0), (convert(
            self.start.x), convert(self.start.y)), CELL_WIDTH // 2)
        pygame.draw.circle(self.screen, (0, 255, 0), (convert(
            self.final.x), convert(self.final.y)), CELL_WIDTH // 2)

    def draw_dijkstra(self):
        self.path_dijkstra = []
        self.dijkstra_time = time.time()
        self.evaluate_dijkstra()
        self.reset_board()
        print('\nDijkstra')
        for curr in self.dijkstra_generated:
            pygame.draw.circle(self.screen, (0, 255, 255),
                               (convert(curr[0]), convert(curr[1])), 3)
            pygame.display.update()
        for curr in self.path_dijkstra:
            print(curr.x, curr.y)
            pygame.draw.circle(self.screen, (255, 0, 255),
                               (convert(curr.x), convert(curr.y)), 5)
            pygame.display.update()
        print("\nCost:", self.path_dijkstra[-1].cost)
        print("Time: ", self.dijkstra_time)

    def draw_best_first_search(self):
        self.best_first_path = []
        self.best_first_time = time.time()
        self.evaluate_best_first()
        self.reset_board()
        print('\nBest First Search')
        for curr in self.best_first_generated:
            pygame.draw.circle(self.screen, (0, 255, 255),
                               (convert(curr[0]), convert(curr[1])), 3)
            pygame.display.update()
        for curr in self.best_first_path:
            print(curr.x, curr.y)
            pygame.draw.circle(self.screen, (255, 255, 0),
                               (convert(curr.x), convert(curr.y)), 5)
            pygame.display.update()
        print("\nCost:", self.best_first_path[-1].cost)
        print("Time: ", self.best_first_time)

    def draw_a_star(self, over=False):
        self.path_a_star = []
        self.a_star_time = time.time()
        self.evaluate_a_star(over)
        self.reset_board()
        print('\nA Star Search')
        for curr in self.a_star_generated:
            pygame.draw.circle(self.screen, (0, 255, 255),
                               (convert(curr[0]), convert(curr[1])), 3)
            pygame.display.update()
        for curr in self.path_a_star:
            print(curr.x, curr.y)
            pygame.draw.circle(self.screen, (255, 255, 255),
                               (convert(curr.x), convert(curr.y)), 5)
            pygame.display.update()
        print("\nCost:", self.path_a_star[-1].cost)
        print("Time: ", self.a_star_time)


# 0 14
# 29 14
# 10 0 10 20
start = [int(x) for x in input().split()]
end = [int(x) for x in input().split()]
app = App(Coord(start[0], start[1]), Coord(end[0], end[1]))
