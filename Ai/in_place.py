import numpy as np
import random

class InPlace:
    def __init__(self, V, enviroment, policy) -> None:
        self.V = V
        self.enviroment = enviroment
        self.policy = policy
        self.ai_flag = True
        self.better_policy = {}


    def value_iteration(self, states, gamma, theta):
        V = dict()
        policy = dict()


        for state in states:
            V[hash(state)] = 0
            policy[hash(state)] = 'STAY'

        while True:
            delta = 0
            for state in states:
                best_value_function = []
                best_actions = []
                for action in self.enviroment.get_possible_actions(state):
                    v = 0
                    next_states = self.enviroment.get_next_states(state, action)
                    for next_state in next_states:
                        try:
                            value_function = V[next_state]
                            reward = self.enviroment.get_reward(next_state)
                            pass_probability = next_states[next_state]
                            v += pass_probability * (reward + gamma * value_function)
                        except KeyError:
                            pass

                    best_value_function.append(v)
                    best_actions.append(v)

                best_value = max(best_value_function)
                delta = max(delta, np.abs(best_value - V[hash(state)]))
                best_action = np.argmax(best_actions)
                better_action = self.enviroment.get_possible_actions(state)[best_action]
                V[hash(state)] = best_value
                policy[hash(state)] = better_action

            if delta < theta:
                break
        
        self.better_policy = policy
        self.V = V

    def move_paddle(self, current_state, left = True):
        try:
            action = self.better_policy[hash(current_state)][1]
            if action == 'STAY':
                return 2
            if action == 'UP':
                return True
            if action == 'DOWN':
                return False

        except KeyError:
             lst = [True, 2 , False]
             return random.choice(lst)
        
