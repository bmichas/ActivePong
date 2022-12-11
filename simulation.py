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
            next_state, reward, done, _  = self.env.step(action)
            print(state)
            self.env.draw()
            # state = (ball_state, ball_velocity, left_paddle_state)
            action = self.agent.update(state, action, reward, next_state)
            state = next_state
            if self.env.left_score >= self.win_score:
                self.left_win_rate += 1
            elif self.env.right_score >= self.win_score:
                self.right_win_rate += 1

            if done:
                pygame.display.update()
                break

            pygame.display.update()