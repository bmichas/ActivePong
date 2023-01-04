from collections import defaultdict
import random
import math
import numpy as np


class MCTS:
    def __init__(self, get_legal_actions, env, simulation_no) -> None:
        self.get_legal_actions = get_legal_actions
        self.env = env
        self.tree = defaultdict(self.def_value)
        self.n = 0
        self.t = 0
        self.simulation_no = simulation_no
        self.name = 'MCTS'


    def mcts_reset(self):
        self.tree = defaultdict(self.def_value)
        self.n = 0
        self.t = 0


    def def_value(self):
        return {'parent': None,
                'action': None,
                'state': None,
                'childs': [],
                't': 0,
                'n': 0,
                'ucb1': float('inf')}
        

    def _gen_ball_state(self, ball_state, ball_vel_state):
        ball_state_x = ball_state[0]
        ball_state_y = ball_state[1]
        ball_vel_x = ball_vel_state[0]
        ball_vel_y = ball_vel_state[1]
        if ball_state_y + self.env.ball.height >= self.env.window_height:
            ball_vel_y *= -1
        
        elif ball_state_y <= 0:
            ball_vel_y *= -1
        
        ball_state_x += ball_vel_x
        ball_state_y += ball_vel_y
        return (ball_state_x, ball_state_y), (ball_vel_x, ball_vel_y)


    def _roll_out(self, state, action):
        ball_state = state[0]
        ball_vel_state = state[1]
        left_paddle_state = list(state[2])
        ball_state, ball_vel_state = self._gen_ball_state(ball_state, ball_vel_state)
        next_state = (ball_state, ball_vel_state, tuple(left_paddle_state))
        if action == 'STAY':
            next_state = (ball_state, ball_vel_state, tuple(left_paddle_state))

        if action == 'UP':
            left_paddle_state[1] -= 50
            next_state = (ball_state, ball_vel_state, tuple(left_paddle_state))

        if action == 'DOWN':
            left_paddle_state[1] += 50
            next_state = (ball_state, ball_vel_state, tuple(left_paddle_state))

        return next_state, action


    def _expand(self, state, possible_actions):
        
        self.tree[hash(state)]['state'] = state
        for action in possible_actions:
            next_state, action = self._roll_out(state, action)
            self.tree[hash(state)]['childs'].append(hash(next_state))
            self.tree[hash(next_state)]['state'] = next_state
            self.tree[hash(next_state)]['action'] = action
            self.tree[hash(next_state)]['parent'] = hash(state)


    def _select(self, state, possible_actions):
        is_terminal = False
        path = []
        path.append(hash(state))
        while not is_terminal:
            if self.n == 0:
                self._expand(state, possible_actions)
                self.n += 1

            best_child = self.select_best_child(self.tree[hash(state)]['childs'])
            if not self.tree[best_child]['childs']:
                possible_actions = self.get_legal_actions(state)
                child_state = self.tree[best_child]['state']
                self._expand(child_state, possible_actions)
            
            best_action = self.tree[best_child]['action']
            next_state, action = self._roll_out(state, best_action)
            state = next_state
            path.append(hash(state))
            is_terminal, reward = self.env.get_reward_state(state)
            if is_terminal:
                action = self.tree[path[1]]['action']
                self._back_propagation(path, reward)
                break
            
        return action

    def select_best_child(self, child_list):
        best_ucb1 = float("-inf")
        best_child_list = []
        if not child_list:
            done = False
            while not done:
                best_child_list = list(self.tree.keys())
                best_child = random.choice(best_child_list)
                if self.tree[best_child]['parent'] != None:
                    done = True
                    break

        else:
            for child in child_list:
                if best_ucb1 <= self.tree[child]['ucb1']:
                    best_child_list.append(child)
                    best_ucb1 = self.tree[child]['ucb1']
            best_child = random.choice(best_child_list)

        return best_child


    def _back_propagation(self, path, reward): 
        path.reverse()
        for node in path:
            self.tree[node]['t'] += reward
            self.tree[node]['n'] += 1
            self._update_ucb1(node)


    def _update_ucb1(self, node):
        t = self.tree[node]['t']
        n = self.tree[node]['n']
        parent = self.tree[node]['parent']
        parent_n = self.tree[parent]['n']
        if parent_n == 0:
            parent_n += 1

        vi = t/n
        ucb1 = vi + (2 * math.sqrt(np.log(parent_n) / n))
        self.tree[node]['ucb1'] = ucb1


    def best_action(self, state):
        possible_actions = self.get_legal_actions(state)
        # print('=', state)
        ball_vel_x = state[1][0]
        if len(possible_actions) == 0 or ball_vel_x > 0:
            return None

        else:
            for _ in range(self.simulation_no):
                chosen_action = self._select(state, possible_actions)

        self.mcts_reset()
        return chosen_action



