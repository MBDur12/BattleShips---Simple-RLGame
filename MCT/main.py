import sys
sys.path.append("./Battleships")
import Game
import Board
from AI import AI
from random import randint

SHIPS = {"Patrol Boat": 2, "Submarine": 3, "Destroyer": 3, "Battleship": 4, "Carrier": 5}


def random_move(board):
    guess = tuple([randint(0, board.width-1), randint(0, board.width-1)])
    while board.board[guess[0], guess[1]] != 0:
        guess = tuple([randint(0, board.width-1), randint(0, board.width-1)])

    return guess

def main():
    # Setup game envs for computer and player
    comp_board = Board.Board(10)
    player_board = Board.Board(10)

    for k,v in SHIPS.items():
        comp_board.place_ship(k, v)
        player_board.place_ship(k, v)


    comp_env = Game.Game(comp_board, player_board)
    player_env = Game.Game(player_board, comp_board)
    
    # Setup a "computer player" to run through MCTS before returning a move
    computer = AI(comp_env, 10) # class that wraps MCTS functionality.

    # Start game loop, running until either environment is over.
    comp_done, player_done = False, False
    while not comp_done and not player_done:
        plyr_state, player_done = computer.move()
        cmp_state, comp_done = player_env.step(random_move(player_env.opp_board))

    
        print("Player's board:")
        player_env.own_board.display()
        print("Computer's board:")
        comp_env.own_board.display()
    
    if comp_done:
        print("Computer lost")
    else:
        print("Player lost")
    
    









if __name__ == "__main__":
    main()