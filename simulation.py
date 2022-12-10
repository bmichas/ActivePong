import pygame

class Simulation:
    def __init__(self, fps ,game, win_score) -> None:
        self.fps = fps
        self.enviroment = game
        self.win_score = win_score
        self.clock = pygame.time.Clock()
        self.left_win_rate = 0
        self.right_win_rate = 0


    def run(self):
        run = True
        while run:
            self.clock.tick(self.fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break
     
            # PRINT TO CHECK IF NEXT STATES ARE WORKING
            # current_state = self.enviroment.get_current_state()
            # actions = self.enviroment.get_possible_actions(current_state)
            # for action in actions:
            #     next_states = self.enviroment.get_next_states(current_state, action)
            #     print("State: " + str(current_state) + " action: " + str(action) + " " + "list of possible next states: ", str(next_states))

            
            self.enviroment.step()
            # print(self.enviroment.get_reward())
            self.enviroment.draw()
            won = False
            if self.enviroment.left_score >= self.win_score:
                self.left_win_rate += 1
                won = True
            elif self.enviroment.right_score >= self.win_score:
                self.right_win_rate += 1
                won = True

            if won:
                self.enviroment.reset()
                print(self.left_win_rate, ':', self.right_win_rate)
                run = False

            pygame.display.update()