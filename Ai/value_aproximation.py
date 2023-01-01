
from collections import defaultdict
import numpy as np
import random
import math


class ValueAproxAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.weights = [0.5, 0.5, 0.5]
        self.name = 'ValueAproxAgent'


    def get_qvalue(self, state, action):
        return self._qvalues[state][action]


    def set_qvalue(self, state, action, value):
        self._qvalues[state][action] = value


    def count_f_position(self, state):
        paddle_position = list(state[2])
        ball_position = list(state[0])
        return math.dist(paddle_position, ball_position)


    def count_f_velocity(self, state):
        factor = 1
        ball_vel = list(state[1])
        if ball_vel[0] < 0:
            factor = -1

        return factor

    def count_f_pallet(self, state):
        return state[2][0]

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

    def update(self, state, action, reward, next_state, position):
        gamma = self.discount
        learning_rate = self.alpha
        value_function = [self.count_f_position(position), self.count_f_velocity(position), self.count_f_pallet(position)]
        value_function = self.norm(value_function)
        updated_qvalue = 0
        for i in range(len(value_function)):
            updated_qvalue += self.weights[i] * value_function[i]

        self.set_qvalue(state, action, updated_qvalue)
        difference = (reward + gamma * self.get_value(next_state)) - self.get_qvalue(state, action)
        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * difference * value_function[i]
        
        next_action = self.get_action(next_state)
        return next_action


    def get_best_action(self, state):
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return None

        actions_qvalues_dict = {}
        for action in possible_actions:
            actions_qvalues_dict[action] = self.get_qvalue(state, action)

        best_q_avlue = max(list(actions_qvalues_dict.values()))
        best_actions = []
        for action in actions_qvalues_dict:
            if actions_qvalues_dict[action] == best_q_avlue:
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
