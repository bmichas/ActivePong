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

    def _draw_score(self):
        left_score_text = self.SCORE_FONT.render(f"{self.left_score}", 1, self.WHITE)
        right_score_text = self.SCORE_FONT.render(f"{self.right_score}", 1, self.WHITE)
        self.window.blit(left_score_text, (self.window_width // 2 - 50, 50))
        self.window.blit(right_score_text, (self.window_width // 2 + 25, 50))


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


    def step(self):
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
            current_state = self.get_current_state()
            move = self.ai_left.move_paddle(current_state, False)
            self.move_left_paddle(move)
        
        else:
            if keys[pygame.K_w]:
                self.move_left_paddle(up=True)

            if keys[pygame.K_s]:
                self.move_left_paddle(up=False)

        
        self.ball.move()
        self._handle_collision(self.ball, self.left_paddle, self.right_paddle)
        
        # print('BALL:', self.ball.x, self.ball.y, self.ball.x_vel, self.ball.y_vel)
        # print('LeftPaddle:', self.left_paddle.x, self.left_paddle.y, self.left_hit_count)
        # print('RightPaddle:', self.right_paddle.x, self.right_paddle.y, self.right_hit_count)
        
        # same position handling
        if prev_left_position == self.left_paddle_position or prev_right_position == self.right_paddle_position:
            self.counter += 1
            if self.counter == 200:
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
            
    
    def _get_all_ball_states(self):
        all_ball_position = []
        for multiplayer_y in range(self.window_height // self.block_size + 1):
            for multiplayer_x in range(self.window_width // self.block_size + 1):
                ball_coordinates = (self.block_size * multiplayer_x, self.block_size * multiplayer_y)
                all_ball_position.append(ball_coordinates)

        return all_ball_position


    def _get_all_paddle_states(self, paddle):
        all_paddle_position = []
        for multiplayer_y in range(self.window_height // self.block_size - PADDLE_HEIGHT // self.block_size + 1):
            pallet_coordinates = (paddle.x, self.block_size * multiplayer_y)
            all_paddle_position.append(pallet_coordinates)
        
        return all_paddle_position

    
    def get_all_states(self):
        ball_positions = self._get_all_ball_states()
        left_paddle_positions = self._get_all_paddle_states(self.left_paddle)
        right_paddle_positions = self._get_all_paddle_states(self.right_paddle)
        states = []
        reverse_hash_states = {}
        states_tree = {}
        for ball_position in ball_positions:
            for left_paddle_position in left_paddle_positions:
                for right_paddle_position in right_paddle_positions:
                    state = (ball_position, right_paddle_position, left_paddle_position)
                    reverse_hash_states[hash(state)] = state
                    states_tree[hash(state)] = 0
                    states.append(state)
        
        self.states = states
        self.reverse_hash_states = reverse_hash_states
        self.tree = states_tree
        return states, states_tree
            

    def _get_all_paddle_action(self, paddle):
        possible_paddle_action = ['STAY']
        if paddle[1] >= 0 and paddle[1] != self.window_height - self.right_paddle.height:
            possible_paddle_action.append('DOWN')

        if paddle[1] <= self.window_height - self.right_paddle.height and paddle[1] != 0:
            possible_paddle_action.append('UP')
            
        return possible_paddle_action 


    def _get_all_ball_action(self, ball_y):
        if ball_y == 0:
            return ['LEFT','LEFT_DOWN', 'RIGHT', 'RIGHT_DOWN']
        
        elif ball_y + BALL_HEIGHT == self.window_height :
            return ['LEFT','LEFT_UP', 'RIGHT', 'RIGHT_UP']
        
        else:
            return ['LEFT','LEFT_DOWN', 'LEFT_UP', 'RIGHT', 'RIGHT_DOWN', 'RIGHT_UP']
        
    
    def _gen_possible_actions(self, state):
        """
            state = [(ball position), (right paddle), (left paddle)]
            possible ball movement:
                LEFT UP: x_vel < 0 and y_vel < 0
                LEFT : x_vel < 0 and y_vel == 0
                LEFT DOWN : x_vel < 0 and y_vel > 0
                RIGHT UP : x_vel > 0 and y_vel < 0
                RIGHT : x_vel > 0 and y_vel == 0
                RIGHT DOWN : x_vel > 0 and y_vel > 0
            possible right paddle movement:
                UP: Y.position > 0
                STAY: can always stay
                DOWN: Y.position < window.height
                                                    """
        
        ball_position = state[0]
        right_paddle_position = state[1]

        ball_movements = self._get_all_ball_action(ball_position[1])
        possible_right_paddle_movement = self._get_all_paddle_action(right_paddle_position)
        possible_action = []
        for right_paddle_movement in possible_right_paddle_movement:
            for ball_movement in ball_movements:
                move = (ball_movement, right_paddle_movement)
                possible_action.append(move)

        return possible_action
       

    def _gen_next_states(self, state, action):
        ball_action_dic = {
            'LEFT_UP': (self.ball.x_vel, self.ball.y_vel),
            'LEFT': (self.ball.x_vel, 0),
            'LEFT_DOWN': (self.ball.x_vel, self.ball.y_vel),
            'RIGHT_UP': (self.ball.x_vel, self.ball.y_vel),
            'RIGHT': (self.ball.x_vel, 0),
            'RIGHT_DOWN':(self.ball.x_vel, self.ball.y_vel)
        }
        paddle_action_dic = {
            'UP': (0, self.right_paddle.velocity),
            'STAY': (0, 0),
            'DOWN': (0, -self.right_paddle.velocity)        
        }
        
        # Position of objects
        ball_state = state[0]
        right_paddle_state = state[1]
        left_paddle_state = state[2]
        
        # Action of objects
        ball_action = action[0]
        right_paddle_action = action[1]
        
        ball_move = ball_action_dic[ball_action]
        right_paddle_move = paddle_action_dic[right_paddle_action]
        
        next_ball_state = (ball_state[0] + ball_move[0], ball_state[0] + ball_move[0])
        next_right_paddle_state = (right_paddle_state[0] + right_paddle_move[0], right_paddle_state[1] + right_paddle_move[1])

        possible_left_paddle_movement = self._get_all_paddle_action(left_paddle_state)
        possible_next_states = {}
        for left_paddle_action in possible_left_paddle_movement:
            left_paddle_move = paddle_action_dic[left_paddle_action]
            next_left_paddle_state = (left_paddle_state[0] + left_paddle_move[0], left_paddle_state[1] + left_paddle_move[1])
            possible_next_state = (next_ball_state, next_right_paddle_state, next_left_paddle_state)
            possible_next_states[hash(possible_next_state)] = random.random()

        return possible_next_states

        
    def get_tree(self, states):
        counter = 0
        for state in states:
            actions = self._gen_possible_actions(state)
            possible_actions = {}
            for action in actions:
                possible_actions[action] = 0
            self.tree[hash(state)] = possible_actions

            for action in actions:
                possible_next_states = self._gen_next_states(state, action)
                self.tree[hash(state)][action] = possible_next_states
                counter += 1
                # print("Counter:" + str(counter) + "State: " + str(state) + " action: " + str(action) + " " + "list of possible next states: ", str(possible_next_states))

        print('STATE -> ACTION -> STATE: ', counter)
        return self.tree

    def get_policy(self, states):
        policy = dict()
        for s in states:
            actions = self.get_possible_actions(s)
            action_prob = 1 / len(actions)
            policy[hash(s)] = dict()
            for a in actions:
                policy[hash(s)][a] = action_prob
        
        return policy

    
    def get_possible_actions(self, state):
        return tuple(self.tree.get(hash(state), {}).keys())


    def get_next_states(self, state, action):
        return self.tree[hash(state)][action]


    def get_current_state(self):
        return ((self.ball.x, self.ball.y), (self.right_paddle.x, self.right_paddle.y), (self.left_paddle.x, self.left_paddle.y))


    def get_reward(self, state):
        state = self.reverse_hash_states[state]
        ball_x = state[0][0]
        ball_y = state[0][1]
        left_paddle_x = state[2][0]
        left_paddle_y = state[2][1]

        if ball_y >= left_paddle_y and ball_y  < left_paddle_y + self.left_paddle.height:
                if ball_x - self.ball.width <= left_paddle_x:
                    return 2
        elif ball_x == 0:
            return -1
        elif ball_x == self.window_width:
            return 3

        return 0


    """REWARD DURING PLAYING"""
    # def get_reward(self):
    #     if self.left_hit_count_prev < self.left_hit_count:
    #         self.left_hit_count_prev = self.left_hit_count
    #         return 1
    #     elif self.right_score_prev < self.right_score:
    #         self.right_score_prev = self.right_score
    #         return -1
    #     elif self.left_score_prev < self.left_score:
    #         if self.left_score == self.win_score:
    #             return 5
    #         self.left_score_prev = self.left_score
    #         return 2
    #     else:
    #         return 0

    def reset(self):
        self.ball.reset()
        self.left_paddle.reset()
        self.right_paddle.reset()
        self.left_score = 0
        self.right_score = 0
        self.left_hit_count = 0
        self.right_hit_count = 0
        self.counter = 0