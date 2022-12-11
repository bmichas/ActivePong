import pygame
from simulation import Simulation
from pong import Pong
from Ai import SimpleAi
from Ai import InPlace
from Ai import RandomAi
from Ai import DQLearningAgent
from Ai import ExpectedSARSAAgent
from Ai import SARSALambdaAgent
from Ai import QLearningAgent
from Ai import SARSAAgent



LEARN_EPOCH = 20000
GAME_EPOCH = 100
WIN_SCORE = 1
WIDTH, HEIGHT = 350, 350
# WIDTH, HEIGHT = 650, 650
FPS = 100000
VELOCITY = 50


env = Pong(WIDTH, HEIGHT, VELOCITY, WIN_SCORE)
random_ai = RandomAi()
agent_sarsa = SARSAAgent(alpha = 0.5, epsilon = 0.25, discount = 0.99, get_legal_actions = env.get_possible_actions)
agent_q = QLearningAgent(alpha = 0.5, epsilon = 0.25, discount = 0.99,get_legal_actions = env.get_possible_actions)
agent_expected = ExpectedSARSAAgent(alpha = 0.1, epsilon = 0.1, discount = 0.99, get_legal_actions = env.get_possible_actions)
agent_lambda = SARSALambdaAgent(alpha = 0.1, epsilon = 0.1, discount = 0.99, get_legal_actions = env.get_possible_actions, lambda_value = 0.5)
agent_dq = DQLearningAgent(alpha = 0.1, epsilon = 0.1, discount = 1, get_legal_actions = env.get_possible_actions)

AGENT_DICT = {
    agent_sarsa: 0,
    agent_q: 0,
    agent_expected: 0,
    agent_lambda: 0,
    agent_dq: 0
}

def learn_play(env, agent):
    print(agent.name)
    env.set_left_ai(agent)
    env.set_right_ai(random_ai)

    sim1 = Simulation(FPS, env, agent, WIN_SCORE)
    for _ in range(LEARN_EPOCH):
        sim1.run()

    agent.turn_off_learning()
    # sim1.set_fps(10)
    sim1.set_win_score(10)
    env.set_win_score(10)
    for _ in range(GAME_EPOCH):
        sim1.reset_win_rate()
        sim1.run()
        AGENT_DICT[agent] = str(sim1.left_win_rate) + ' : ' + str(sim1.right_win_rate)

    print(str(sim1.left_win_rate) + ' : ' + str(sim1.right_win_rate))

for key in AGENT_DICT.keys():
    learn_play(env, key)


for agent in AGENT_DICT:
    print(agent.name, 'SCORE ', AGENT_DICT[agent])



        





