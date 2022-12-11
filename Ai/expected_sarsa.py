import random
from collections import defaultdict


class ExpectedSARSAAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.name = 'ExpectedSARSAAgent'


    def get_qvalue(self, state, action):
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        self._qvalues[state][action] = value


    def get_value(self, state):
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return 0.0

        q_values = []
        for action in possible_actions:
            q_values.append(self.get_qvalue(state, action))

        max_value = max(q_values)
        return max_value


    def update(self, state, action, reward, next_state):
        gamma = self.discount
        learning_rate = self.alpha
        epsilon = self.epsilon

        def get_greedy(max_value):
            greedy = 0
            for next_action in self.get_legal_actions(next_state):
                if self.get_qvalue(next_state, next_action) == max_value:
                    greedy += 1
            return greedy


        sum_pi  = 0
        max_value = self.get_value(next_state)
        greedy = get_greedy(max_value)
        non_greedy_prob = epsilon / len(self.get_legal_actions(next_state))
        greedy_prob = ((1 - epsilon) / greedy) + non_greedy_prob
        for next_action in self.get_legal_actions(next_state):
            if self.get_qvalue(next_state, next_action) == max_value:
                sum_pi += greedy_prob * self.get_qvalue(next_state, next_action)

            else:
                sum_pi += non_greedy_prob * self.get_qvalue(next_state, next_action)

        updated_qvalue = (1 - learning_rate) * self.get_qvalue(state, action) + \
                                    (learning_rate*(reward + gamma * sum_pi))
        self.set_qvalue(state, action, updated_qvalue)
        

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