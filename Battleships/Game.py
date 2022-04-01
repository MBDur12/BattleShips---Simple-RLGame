import Board

class Game():
    def __init__(self, own_board, opp_board):
        self.own_board = own_board
        self.opp_board = opp_board
        

    def step(self, coords):
        # Take a shot at the opponent's board
        self.opp_board.hit(coords)
        # return the state of the opp's board and whether the game is over
        done = self.opp_board.is_game_over()
        return self.opp_board, done


        

