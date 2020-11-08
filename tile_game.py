import pygame
from pygame.locals import *
import queue
import copy

pygame.init()
pygame.font.init()

default_font = pygame.font.get_default_font()
font_renderer = pygame.font.Font(default_font, 30)

color = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255),
         (125, 125, 125)]
WIDTH = 300
CELL_WIDTH = WIDTH // 3


class State:

    def __init__(self, b):
        self.board = b
        self.heuristic_val = 0
        self.dist = 100000000
        self.zro = self.set_zro()
        self.prev = None

    def set_zro(self):
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    return i, j

    def __lt__(self, other):
        return True


def check(x, y):
    return 0 <= x < 3 and 0 <= y < 3


def convert(board):
    temp = [x for y in board for x in y]
    res = tuple(temp)
    return res


def print_board(board, heuristic_val):
    for row in board:
        print(row)
    print("Heuristic value: ", heuristic_val)
    print()


class App:

    def __init__(self, s, e):
        self.screen = pygame.display.set_mode((WIDTH, WIDTH))
        self.running = True
        self.start = s
        self.final = e
        self.steps = self.eval()

    def run(self):
        while self.running:
            self.events()

    def eval(self):

        self.start.heuristic_val = self.eval_heuristic(self.start)

        dx = [0, 0, -1, 1]
        dy = [1, -1, 0, 0]
        f = None
        curr = copy.deepcopy(self.start)
        pq = queue.PriorityQueue()
        curr.heuristic_val = self.eval_heuristic(curr)

        curr.dist = 0
        pq.put((curr.heuristic_val, curr))

        visit = dict()
        visit[convert(curr.board)] = curr
        level = 0

        while not pq.empty():
            level += 1
            top = pq.get()
            curr = top[1]
            # print(top[0])
            if curr.heuristic_val == 0:
                f = curr
                break
            x, y = curr.zro
            print("Level: ", level)
            print("***********************************")
            print()
            for i, j in zip(dx, dy):
                nx = x + i
                ny = y + j
                if not check(nx, ny):
                    continue
                s = copy.deepcopy(curr)
                # print(s.board)
                s.board[x][y], s.board[nx][ny] = s.board[nx][ny], s.board[x][y]
                # print(s.board)
                bh = convert(s.board)
                s.dist = 1000000
                if bh in visit:
                    # print(visit[bh])
                    v = visit[bh]
                    s.dist = v.dist
                    s.prev = v.prev
                    s.zro = v.zro
                    s.heuristic_val = v.heuristic_val
                    visit.pop(bh)
                    # print(type(s))
                if curr.dist + 1 < s.dist:
                    s.dist = curr.dist + 1
                    s.zro = (nx, ny)
                    s.heuristic_val = self.eval_heuristic(s)
                    print_board(s.board, s.heuristic_val)
                    s.prev = curr
                    visit[bh] = s
                    # print(visit[bh])
                    pq.put((s.heuristic_val, s))
            print()
            print()
        res = []
        while f.prev is not None:
            res.append(f)
            f = f.prev
        res.append(self.start)
        return res

    def eval_heuristic(self, s):
        score = 0
        curr = s.board
        for i in range(3):
            for j in range(3):
                if curr[i][j] != self.final.board[i][j]:
                    score += 1
        return score

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                # print("event")
                self.update()

    def update(self):
        if not self.steps:
            return
        state = self.steps.pop()
        print_board(state.board, state.heuristic_val)
        self.draw(state)
        pygame.display.update()

    def draw(self, state):
        self.screen.fill((0, 0, 0))
        _y = CELL_WIDTH // 2
        for i in range(3):
            _x = CELL_WIDTH // 2
            for j in range(3):
                if not state.board[i][j] == 0:
                    label = font_renderer.render(str(state.board[i][j]), 1, (255, 255, 255))
                    self.screen.blit(label, (_x, _y))
                    # pygame.draw.rect(self.screen, color[state.board[i][j]],
                    #                  (i * CELL_WIDTH, j * CELL_WIDTH, CELL_WIDTH, CELL_WIDTH))
                _x += CELL_WIDTH
            _y += CELL_WIDTH


s = State([[2, 8, 3], [1, 6, 4], [7, 0, 5]])
e = State([[1, 2, 3], [8, 0, 4], [7, 6, 5]])

app = App(s, e)
app.run()
