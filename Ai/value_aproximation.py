
from collections import defaultdict
import numpy as np
import random
import math


class ValueAproxAgent:
    def __init__(self, alpha, epsilon, discount, width, height, get_legal_actions):
        self.get_legal_actions = get_legal_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.weights = [0.5, 0.5, 0.5, 0.5, 0.5]
        self.name = 'ValueAproxAgent'
        self.board_width = width
        self.board_height = height


    def get_qvalue(self, state, action):
        value_function = self.get_value_function(state, action)
        qvalue = 0
        for i in range(len(value_function)):
            qvalue += self.weights[i] * value_function[i]

        return qvalue


    def count_f_position(self, state, action):

        ball_position = list(state[0])
        if action == "STAY":
            paddle_position = list(state[2])
            return math.dist(paddle_position, ball_position)/self.board_width

        if action == "DOWN":
            paddle_position = list(state[2])
            paddle_position[1] += 50
            return math.dist(paddle_position, ball_position)/self.board_width
        
        if action == "UP":
            paddle_position = list(state[2])
            paddle_position[1] -= 50
            return math.dist(paddle_position, ball_position)/self.board_width

    def count_f_velocity(self, state, action):
        factor = 1
        ball_vel = list(state[1])
        if ball_vel[0] < 0:
            factor = -1

        return factor


    def count_f_pallet(self, state, action):
        if action == "STAY":
            paddle_position = list(state[2])
            return paddle_position[1]/self.board_height

        if action == "DOWN":
            paddle_position = list(state[2])
            paddle_position[1] += 50
            return paddle_position[1]/self.board_height
        
        if action == "UP":
            paddle_position = list(state[2])
            paddle_position[1] -= 50
            return paddle_position[1]/self.board_height


    def get_value(self, state):
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return 0.0

        q_values = []
        for action in possible_actions:
            q_values.append(self.get_qvalue(state, action))

        max_value = max(q_values)
        return max_value


    def norm(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    
    def get_value_function(self, state, action):
        value_function = [self.count_f_position(state, action), self.count_f_pallet(state, action), self.count_f_velocity(state, action), state[0][0]/self.board_width, state[0][1]/self.board_height]
        return self.norm(value_function)


    def update(self, state, action, reward, next_state):
        gamma = self.discount
        learning_rate = self.alpha
        difference = (reward + gamma * self.get_value(next_state)) - self.get_qvalue(state, action)
        value_function = self.get_value_function(state, action)
        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * difference * value_function[i]



    def get_best_action(self, state):
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return None

        actions_qvalues_dict = {}
        for action in possible_actions:
            actions_qvalues_dict[action] = self.get_qvalue(state, action)

        best_q_value = max(list(actions_qvalues_dict.values()))
        best_actions = []
        for action in actions_qvalues_dict:
            if actions_qvalues_dict[action] == best_q_value:
                best_actions.append(action)

        best_action = random.choice(best_actions)
        return best_action

    def get_action(self, state):
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return None

        epsilon = self.epsilon     
        if random.random() >= epsilon:
            chosen_action = self.get_best_action(state)
            
        else:
            chosen_action = random.choice(possible_actions)

        return chosen_action

    def turn_off_learning(self):
        self.epsilon = 0
        self.alpha = 0
