import pygame
import string_solver
import numpy as np

pygame.init()
screen = pygame.display.set_mode((0, 0))
width, height = pygame.display.get_surface().get_size()
run = 1
colors = {"white": (255, 255, 255), "orange": (255, 127, 39), "green": (150, 253, 55), "blue": (0, 162, 132),
          "grey": (64, 64, 64), "black": (0, 0, 0), "red": (255, 0, 0)}

string_solver.load_model((width / 2 - 700, width / 2 - 100), (height / 2 - 100, height / 2 + 100), 256)
mode = 2
x, t, n = np.linspace(0, 10, 501), np.ones(501), mode * np.ones(501)
shift = 800

def animate(frame):
    t_ = frame * t
    pygame.draw.lines(screen, (70, 255, 70), False, string_solver.transform(x, string_solver.predict(x, t_, n)[1]), 3)
    pygame.draw.lines(screen, (255, 70, 70), False, string_solver.transform(x, string_solver.predict(x, t_, n)[0], shift), 3)
    pygame.draw.line(screen, (70, 70, 70), (width / 2 - 2, 10), (width / 2 - 2, height - 10), 5)

f = 0.0
start = False
while run:
    screen.fill((210, 210, 210))
    if start:
        f += 0.01
    animate(f)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                run = False
            if event.key == pygame.K_SPACE:
                if start:
                    start = False
                else:
                    start = True
    pygame.display.update()
    if f > 20:
        f = 0.0
pygame.quit()
