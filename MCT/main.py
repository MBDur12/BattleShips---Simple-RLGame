from Battleships import Game, Board
from AI import AI

def main():
    # Setup game envs for computer and player
    comp_board = Board.Board(10)
    player_board = Board.Board(10)

    comp_env = Game.Game(comp_board, player_board)
    player_env = Game.Game(player_board, comp_board)
    
    # Setup a "computer player" to run through MCTS before returning a move
    computer = AI(comp_env, 10) # class that wraps MCTS functionality.

    # Start game loop, running until either environment is over.
    comp_done, player_done = False, False
    while comp_done and player_done:
        pass
        """
        make computer move, returning new_state, if game is done
        Do the same for the player, taking input from the user.

        Display the status of the boards.
        """
        """
        Once the game loop has ended, return which "player" has won
        """
    









if __name__ == "__main__":
    main()