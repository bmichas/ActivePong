import pygame
import random

class Ball:
    COLOR = WHITE = (255,255,255)

    def __init__(self, x, y, width, height, velocity) -> None:
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.width = width
        self.height = height
        self.x_vel = velocity
        self.y_vel = 0

    
    def draw(self, win):
        pygame.draw.rect(win, self.COLOR, (self.x, self.y, self.width, self.height))

    
    def move(self):
        self.x += self.x_vel
        self.y += self.y_vel

    
    def reset(self):
        y_vel_lst = [-50,0,50]
        self.y_vel = random.choice(y_vel_lst)
        self.x = self.original_x
        self.y = self.original_y    
        self.x_vel *= -1
        self.y_vel = 0