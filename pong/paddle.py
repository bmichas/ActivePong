import pygame


class Paddle:
    COLOR = WHITE = (255,255,255)


    def __init__(self, x, y, width, height, velocity):
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.width = width
        self.height = height
        self.velocity = velocity

    
    def draw(self, win):
        pygame.draw.rect(win, self.COLOR, (self.x, self.y, self.width, self.height))

    
    def move(self, up=True):
        if up:
            self.y -= self.velocity
        elif up == 2:
            self.y = 0
        else:
            self.y += self.velocity


    def reset(self):
        self.x = self.original_x
        self.y = self.original_y