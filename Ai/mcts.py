from collections import defaultdict
import random
import math
import numpy as np


class MCTS:
    def __init__(self, get_legal_actions) -> None:
        self.get_legal_actions = get_legal_actions
        self.tree = defaultdict(self.def_value)
        self.n = 0
        self.t = 0
        self.v = 0
        self.name = 'MCTS'

    def def_value(self):
        return {'parent': None, 
                'position': None,
                'parent_position': None,
                't': 0, 
                'n': 0, 
                'ucb1': float('inf')}


    def _expand(self, state, possible_actions):
        possible_states = []
        ball_state = state[0]
        ball_vel_state = state[1]
        left_paddle_state = list(state[2])
        for action in possible_actions:
            if action == 'STAY':
                next_state = (ball_state, ball_vel_state, tuple(left_paddle_state))
                possible_states.append(next_state)

            if action == 'UP':
                left_paddle_state[1] -= 50
                next_state = (ball_state, ball_vel_state, tuple(left_paddle_state))
                possible_states.append(next_state)

            if action == 'DOWN':
                left_paddle_state[1] += 50
                next_state = (ball_state, ball_vel_state, tuple(left_paddle_state))
                possible_states.append(next_state)

            left_paddle_state = list(state[2])

        for possible_state in possible_states:
            if hash(possible_state) not in self.tree:
                self.tree[hash(possible_state)]['parent'] = hash(state)
                self.tree[hash(possible_state)]['position'] = possible_state[2]
                self.tree[hash(possible_state)]['parent_position'] = state[2]

        return self._select()
        

    def _select(self):
        max_val = float('-inf')
        possible_actions = []
        for child in self.tree:
            if self.tree[child]['ucb1'] >= max_val:
                max_val = self.tree[child]['ucb1']
                possible_actions.append(self.tree[child])

        action = random.choice(possible_actions)
        return self._count_action(action)


    def _roll_out(self, reward):
        self.v += reward
        


    def _back_propagation(self, leaf):
        leaf['t'] += self.v
        leaf['n'] += 1
        parent = leaf['parent']
        self.tree[parent]['t'] += self.v
        self.tree[parent]['n'] += 1
        self._update_ucb1(leaf)


    def _update_ucb1(self, leaf):
        t=leaf['t']
        n=leaf['n']
        parent_n = self.tree[leaf['parent']]['n']
        vi = t/n
        ucb1 = vi + (2 * math.sqrt(np.log(parent_n) / n))
        leaf['ucb1'] = ucb1

    
    def _count_action(self, action):
        y_new = action['position'][1]
        y_old = action['parent_position'][1]
        diff = y_new - y_old
        if diff == 0:
            return 'STAY', action

        if diff > 0:
            return 'DOWN', action

        if diff < 0:
            return 'UP', action


    def get_action(self, state):
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return None
        
        chosen_action, leaf = self._expand(state, possible_actions)
        return chosen_action


    def update(self, state, action, reward, next_state):
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return None

        chosen_action, leaf = self._expand(state, possible_actions) 
        if leaf['n'] == 0:
            self._roll_out(reward)

        self._back_propagation(leaf)
        self.v = 0
        possible_actions = self.get_legal_actions(next_state)
        chosen_action, leaf = self._expand(next_state, possible_actions) 
        return chosen_action


    def turn_off_learning(self):
        pass



