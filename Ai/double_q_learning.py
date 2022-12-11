import random
from collections import defaultdict


class DQLearningAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        self.get_legal_actions = get_legal_actions
        self._qvaluesA = defaultdict(lambda: defaultdict(lambda: 0))
        self._qvaluesB = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.name = 'DQLearningAgent'
        

    def get_qvalue(self, state, action):
        return self._qvaluesA[state][action] + self._qvaluesB[state][action] 


    def update(self, state, action, reward, next_state):
        gamma = self.discount
        learning_rate = self.alpha
        check = random.choice([True, False])
        if check:
            value = self._qvaluesA[state][action] + learning_rate * (reward + gamma * self._qvaluesB[next_state][self.get_best_action(state)] - self._qvaluesA[state][action])
            self._qvaluesA[state][action] = value

        else:
            value = self._qvaluesB[state][action] + learning_rate * (reward + gamma * self._qvaluesA[next_state][self.get_best_action(state)] - self._qvaluesB[state][action])
            self._qvaluesB[state][action] = value
        

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
        if random.random() > epsilon:
            chosen_action = self.get_best_action(state)

        else:
            chosen_action = random.choice(possible_actions)     

        return chosen_action       


    def turn_off_learning(self):
        self.epsilon = 0
        self.alpha = 0
