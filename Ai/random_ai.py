import random
class RandomAi:
    def __init__(self) -> None:
        pass


    def move_paddle(self, current_state, left = True):
        lst = [True, 2, False]
        return random.choice(lst)


    def flag(self, flag = True):
        return flag
    