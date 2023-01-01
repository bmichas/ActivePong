import pygame

class Simulation:
    def __init__(self, fps ,game, agent, win_score) -> None:
        self.fps = fps
        self.env = game
        self.win_score = win_score
        self.agent = agent
        self.clock = pygame.time.Clock()
        self.left_win_rate = 0
        self.right_win_rate = 0

    
    def set_fps(self, fps):
        self.fps = fps


    def set_win_score(self, win_score):
        self.win_score = win_score

    
    def reset_win_rate(self):
        self.left_win_rate = 0
        self.right_win_rate = 0


    def run(self):
        state = self.env.reset()
        done = False
        action = self.agent.get_action(state)
        while not done:
            self.clock.tick(self.fps)
            if self.agent.name == 'ExpectedSARSAAgent':
                action = self.agent.get_action(state)

            next_state, reward, done, _  = self.env.step(action)
            self.env.draw()
            # state = (ball_state, ball_velocity, left_paddle_state)
            if self.agent.name == 'ValueAproxAgent':
                action = self.agent.update(hash(state), action, reward, hash(next_state), next_state) 
                state = next_state

            else:
                action = self.agent.update(hash(state), action, reward, hash(next_state)) 
                state = next_state

            if self.env.left_score >= self.win_score:
                self.left_win_rate += 1
            elif self.env.right_score >= self.win_score:
                self.right_win_rate += 1

            if done:
                pygame.display.update()
                break

            pygame.display.update()


    def run_mcts(self):
        done = False
        state = self.env.reset()
        while not done:
            self.clock.tick(self.fps)
            action = self.agent.best_action(state)
            next_state, reward, done, _  = self.env.step(action)
            self.env.draw()
            state = next_state
            if self.env.left_score >= self.win_score:
                self.left_win_rate += 1

            elif self.env.right_score >= self.win_score:
                self.right_win_rate += 1

            if done:
                pygame.display.update()
                break

            pygame.display.update()