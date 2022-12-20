import pygame
import random
from pong.paddle import Paddle
from pong.ball import Ball
pygame.init()


PADDLE_WIDTH, PADDLE_HEIGHT = 50, 150
BALL_WIDTH, BALL_HEIGHT = 50, 50


class Pong:
    SCORE_FONT = pygame.font.SysFont("Cambria Bold", 100)
    WHITE = (255,255,255)
    BLACK = (0,0,0)
    ALMOST_BLACK = (30, 30, 30)
    

    def __init__(self, window_width, window_height, velocity, win_score):
        self.window = pygame.display.set_mode((window_width, window_height))
        self.window_width = window_width
        self.window_height = window_height
        self.velocity = velocity
        self.win_score = win_score
        self.block_size = 50
        self.left_paddle = Paddle(50, window_height//2 - PADDLE_HEIGHT//2, PADDLE_WIDTH, PADDLE_HEIGHT, velocity)
        self.right_paddle = Paddle(window_width - 50 - PADDLE_WIDTH, window_height//2 - PADDLE_HEIGHT//2, PADDLE_WIDTH, PADDLE_HEIGHT, velocity)
        self.ball = Ball(window_width // 2 - BALL_WIDTH // 2, window_height // 2 - BALL_HEIGHT // 2, BALL_WIDTH, BALL_HEIGHT, velocity)
        self.left_paddle_position = (self.left_paddle.x, self.left_paddle.y)
        self.right_paddle_position = (self.right_paddle.x, self.right_paddle.y)
        self.ai_left_flag = False
        self.ai_right_flag = False
        self.action_tree = {}
        self.next_state_tree = {}
        self.left_score = 0
        self.left_score_prev = 0
        self.right_score = 0
        self.right_score_prev = 0
        self.left_hit_count = 0
        self.left_hit_count_prev = 0
        self.right_hit_count = 0
        self.counter = 0

    def set_left_ai(self, ai):
        self.ai_left_flag = True
        self.ai_left = ai

    def set_right_ai(self, ai):
        self.ai_right_flag = True
        self.ai_right = ai


    def set_win_score(self, win_score):
        self.win_score = win_score


    def _draw_score(self):
        left_score_text = self.SCORE_FONT.render(f"{self.left_score}", 1, self.WHITE)
        right_score_text = self.SCORE_FONT.render(f"{self.right_score}", 1, self.WHITE)
        self.window.blit(left_score_text, (self.window_width // 2 - 100, 50))
        self.window.blit(right_score_text, (self.window_width // 2 + 50, 50))


    def _handle_collision(self, ball, left_paddle, right_paddle):
        def set_vel(paddle):
            ball.x_vel *= -1
            difference_in_y = paddle.y - ball.y
            if difference_in_y == 0:
                ball.y_vel = -1 * self.velocity

            elif difference_in_y == -50:
                ball.y_vel = 0

            elif difference_in_y == -100:
                ball.y_vel = self.velocity

        # Collision with ball and board
        if ball.y + ball.height >= self.window_height:
            ball.y_vel *= -1
        
        elif ball.y <= 0:
            ball.y_vel *= -1

        # Collision with paddles
        if ball.x_vel < 0:
            # Left
            if ball.y >= left_paddle.y and ball.y < left_paddle.y + left_paddle.height:
                if ball.x - ball.width <= left_paddle.x:
                    set_vel(left_paddle)
                    self.left_hit_count_prev = self.left_hit_count
                    self.left_hit_count += 1

        else:
            # right
            if ball.y >= right_paddle.y and ball.y < right_paddle.y + right_paddle.height:
                if ball.x + ball.width >= right_paddle.x:
                    set_vel(right_paddle)
                    self.right_hit_count += 1


    def draw(self, draw_score = True):
        self.window.fill(self.BLACK)
        for x in range(0, self.window_width, self.block_size):
            for y in range(0, self.window_height, self.block_size):
                rect = pygame.Rect(x, y, self.block_size, self.block_size)
                pygame.draw.rect(self.window, self.ALMOST_BLACK, rect, 1)

        if draw_score:
            self._draw_score()

        self.ball.draw(self.window)
        paddles = [self.left_paddle, self.right_paddle]
        for paddle in paddles:
            paddle.draw(self.window)
        
        self.ball.draw(self.window)


    def move_left_paddle(self, up=True):
        if up and self.left_paddle.y - self.velocity < 0:
            return False
        
        if not up and self.left_paddle.y + PADDLE_HEIGHT >= self.window_height:
            return False

        if up == 2:
            return 2
  
        self.left_paddle.move(up)
        return True
        

    def move_right_paddle(self, up=True):
        if up and self.right_paddle.y - self.velocity < 0:
            return False
        
        if not up and self.right_paddle.y + PADDLE_HEIGHT  >= self.window_height:
            return False

        if up == 2:
            return 2

        self.right_paddle.move(up)
        return True


    def step(self, action):
        keys = pygame.key.get_pressed()
        
        # UP == True, if UP==2: stay
        prev_left_position = self.left_paddle_position
        prev_right_position = self.right_paddle_position
        
        if self.ai_right_flag:
            current_state = self.get_current_state()
            move = self.ai_right.move_paddle(current_state, False)
            self.move_right_paddle(move)
        
        else:
            if keys[pygame.K_UP]:
                self.move_right_paddle(up=True)

            if keys[pygame.K_DOWN]:
                self.move_right_paddle(up=False)
        
        
        if self.ai_left_flag:
            if action == 'STAY':
                self.move_left_paddle(2)
            if action == 'UP':
                self.move_left_paddle(True)
            if action == 'DOWN':
                self.move_left_paddle(False)
            
        else:
            if keys[pygame.K_w]:
                self.move_left_paddle(up=True)

            if keys[pygame.K_s]:
                self.move_left_paddle(up=False)

        
        self.ball.move()
        self._handle_collision(self.ball, self.left_paddle, self.right_paddle)
        
        # same position handling
        if prev_left_position == self.left_paddle_position or prev_right_position == self.right_paddle_position:
            self.counter += 1
            if self.counter == 500:
                self.ball.reset()
                self.left_paddle.reset()
                self.right_paddle.reset()
                self.counter = 0

        
        if self.ball.x == 0:
            self.ball.reset()
            self.left_paddle.reset()
            self.right_paddle.reset()
            self.right_score_prev = self.right_score
            self.right_score += 1
        
        elif self.ball.x == self.window_width:
            self.ball.reset()
            self.left_paddle.reset()
            self.right_paddle.reset()
            self.left_score_prev = self.left_score
            self.left_score += 1

        return self.get_current_state(), self.get_reward(), self.is_terminal(), None

            
    def get_possible_actions(self, state):    
        return ['STAY', 'DOWN', 'UP']
        

    def get_current_state(self):
        state = ((self.ball.x, self.ball.y), (self.ball.x_vel, self.ball.y_vel), (self.left_paddle.x, self.left_paddle.y))
        return state


    def get_reward(self):
        if self.left_hit_count_prev < self.left_hit_count:
            self.left_hit_count_prev = self.left_hit_count
            return 2
        # elif self.right_score_prev < self.right_score:
        #     self.right_score_prev = self.right_score
        #     return -1
        # elif self.left_score_prev < self.left_score:
        #     if self.left_score == self.win_score:
        #         return 5
        #     self.left_score_prev = self.left_score
        #     return 2
        else:
            return 0


    def is_terminal(self):
        if self.left_score == self.win_score or self.right_score == self.win_score:
            return True

        return False


    def reset(self):
        self.ball.reset()
        self.left_paddle.reset()
        self.right_paddle.reset()
        self.left_score = 0
        self.right_score = 0
        self.left_hit_count = 0
        self.right_hit_count = 0
        self.counter = 0
        return self.get_current_state()