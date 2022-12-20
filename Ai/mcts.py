class MCTS:
    def __init__(self) -> None:
        pass

    def select(self):
        pass


    def expand(self):
        pass


    def roll_out(self):
        pass


    def back_propagation(self):
        pass


class Node:
    def __init__(self) -> None:
        self.t_value = 0
        self.n_value = 0


    def set_n_value(self):
        self.n_value += 1

    
    def set_t_value(self, t):
        self.n_value += t
