class SimpleAi:
    def __init__(self) -> None:
        pass

    def move_paddle(self, current_state, left = True):
        ball = current_state[0]
        if left:
            paddle = current_state[2]

        else:
            paddle = current_state[1]

        ball_y = ball[1]    
        paddle_y = paddle[1]
        ratio = paddle_y - ball_y
        if abs(ratio) == 50 or abs(ratio) == 100 or abs(ratio) == 150:
            return 2
        else: 
            if ratio < 0:
                return False
            else:
                return True


    def flag(self, flag = True):
        return flag

    