import pygame
from pygame.locals import *
import copy

pygame.init()
pygame.font.init()

clock = pygame.time.Clock()
WIDTH = 600

dx = [-1, 0, 1]


class Coord:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)

    def __eq__(self, other):
        if not isinstance(other, Coord):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        if not isinstance(other, Coord):
            return NotImplemented
        return self.x != other.x or self.y != other.y


def eval_heuristic(curr, final):
    return (curr.x - final.x) * (curr.x - final.x) + (curr.y - final.y) * (curr.y - final.y)


class App:

    def __init__(self, s, e):
        self.screen = pygame.display.set_mode((WIDTH, WIDTH))
        self.running = True
        self.start = s
        self.final = e
        self.path = []
        self.evaluate()
        self.run()

    def evaluate(self):
        curr = self.start
        while curr != self.final:
            h = eval_heuristic(curr, self.final)
            maxm = curr
            maxm_h = h
            for i in dx:
                for j in dx:
                    if i == j == 0:
                        continue
                    temp = copy.deepcopy(curr)
                    temp.x = curr.x + i
                    temp.y = curr.y + j
                    sh = eval_heuristic(temp, self.final)
                    if sh < maxm_h:
                        if maxm:
                            maxm = temp
                            if maxm_h >= 0:
                                maxm_h = sh
            self.path.append(maxm)
            curr = copy.deepcopy(maxm)

    def run(self):
        while self.running:
            clock.tick(10)
            pygame.draw.circle(self.screen, (255, 0, 0), (self.start.x, self.start.y), 3)
            pygame.draw.circle(self.screen, (0, 255, 0), (self.final.x, self.final.y), 3)
            pygame.display.update()
            self.events()

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                # print("event")
                self.update()

    def update(self):
        self.draw()

    def draw(self):
        for i in self.path:
            pygame.draw.circle(self.screen, (0, 0, 255), (i.x, i.y), 3)
            pygame.display.update()


# 50 50
start = [x for x in input().split()]
# 400 450
end = [x for x in input().split()]
app = App(Coord(start[0], start[1]), Coord(end[0], end[1]))
