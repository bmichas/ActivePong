import pygame
from simulation import Simulation
from pong import Pong
from Ai import RandomAi
from Ai import DQLearningAgent
from Ai import ExpectedSARSAAgent
from Ai import SARSALambdaAgent
from Ai import QLearningAgent
from Ai import SARSAAgent



LEARN_EPOCH = 100
GAME_EPOCH = 1
WIN_SCORE = 1
WIDTH, HEIGHT = 350, 350
WIDTH, HEIGHT = 750, 550
FPS = 1
VELOCITY = 50


env = Pong(WIDTH, HEIGHT, VELOCITY, WIN_SCORE)
random_ai = RandomAi()
agent_sarsa = SARSAAgent(alpha = 0.5, epsilon = 0.4, discount = 0.99, get_legal_actions = env.get_possible_actions)
agent_q = QLearningAgent(alpha = 0.5, epsilon = 0.4, discount = 0.99,get_legal_actions = env.get_possible_actions)
agent_expected = ExpectedSARSAAgent(alpha = 0.4, epsilon = 0.1, discount = 0.99, get_legal_actions = env.get_possible_actions)
agent_lambda = SARSALambdaAgent(alpha = 0.4, epsilon = 0.1, discount = 0.99, get_legal_actions = env.get_possible_actions, lambda_value = 0.5)
agent_dq = DQLearningAgent(alpha = 0.4, epsilon = 0.1, discount = 1, get_legal_actions = env.get_possible_actions)

AGENT_LIST = [agent_sarsa, agent_q, agent_expected, agent_lambda, agent_dq]

def learn_play(env, agent):
    print(agent.name)
    env.set_left_ai(agent)
    env.set_right_ai(random_ai)

    sim1 = Simulation(FPS, env, agent, WIN_SCORE)
    sim1.set_win_score(WIN_SCORE)
    env.set_win_score(WIN_SCORE)
    for _ in range(LEARN_EPOCH):
        sim1.run()

    agent.turn_off_learning()
    sim1.set_fps(50)
    sim1.set_win_score(10)
    env.set_win_score(10)
    sim1.reset_win_rate()
    for _ in range(GAME_EPOCH):
        sim1.run()

    print(str(sim1.left_win_rate) + ' : ' + str(sim1.right_win_rate))
    for qvalue in agent._qvalues:
        print(agent._qvalues[qvalue])

for agent in AGENT_LIST:
    learn_play(env, agent)



        





