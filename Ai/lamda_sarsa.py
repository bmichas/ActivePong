import random
from collections import defaultdict


class SARSALambdaAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions, lambda_value):
        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self._evalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.lambda_value = lambda_value
        self.name = 'SARSALambdaAgent'


    def get_qvalue(self, state, action):
        return self._qvalues[state][action]


    def set_qvalue(self, state, action, value):
        self._qvalues[state][action] = value


    def reset(self):
        self._evalues = defaultdict(lambda: defaultdict(lambda: 0))


    def get_value(self, state):
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return 0.0

        values = []
        for action in possible_actions:
            values.append(self.get_qvalue(state, action))

        return max(values)


    def update(self, state, action, reward, next_state):
        gamma = self.discount
        learning_rate = self.alpha
        next_action = self.get_action(next_state)
        delta = reward + gamma * self.get_qvalue(next_state, next_action) - self.get_qvalue(state, action)
        self._evalues[state][action] += 1
        for lambda_state in self._evalues:
            for lambda_action in self.get_legal_actions(lambda_state):
                value = self.get_qvalue(lambda_state, lambda_action) + learning_rate * delta * self._evalues[lambda_state][lambda_action]
                self.set_qvalue(lambda_state, lambda_action, value)
                self._evalues[lambda_state][lambda_action] = gamma * self.lambda_value * self._evalues[lambda_state][lambda_action]
        
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
        if random.random() > epsilon:
            chosen_action = self.get_best_action(state)

        else:
            chosen_action = random.choice(possible_actions)

        return chosen_action     


    def turn_off_learning(self):
        self.epsilon = 0
        self.alpha = 0

    def display_qvalues(self):
        for s in self._qvalues:
            print("State: " + str(s) + " " + str(self._qvalues[s]))