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

def run_game(games_lost):
    # Setup game envs for computer and player
    comp_board = Board.Board(10)
    player_board = Board.Board(10)

    for k,v in SHIPS.items():
        comp_board.place_ship(k, v)
        player_board.place_ship(k, v)


    comp_env = Game.Game(comp_board, player_board)
    player_env = Game.Game(player_board, comp_board)
    
    # Setup a "computer player" to run through MCTS before returning a move
    computer = AI(comp_env, 60) # class that wraps MCTS functionality.

    # Start game loop, running until either environment is over.
    comp_done, player_done = False, False
    while not comp_done and not player_done:
        plyr_state, player_done = computer.move()
        cmp_state, comp_done = player_env.step(random_move(player_env.opp_board))

    
        #print("Player's board:")
        #player_env.own_board.display()
        #print("Computer's board:")
        #comp_env.own_board.display()
    
    if comp_done:
        #print("Computer lost")
        games_lost.append([player_board, comp_board])
        return False
    else:
        #print("Player lost")
        return True
    
    
def stats():
    """
    Record games when computer loses: to track why. Passed in as argument to run_game()
    """
    games_lost = []

    game_count = 100
    win_count = 0
    for game in range(game_count):
        game_result = run_game(games_lost)
        if game_result == True:
            win_count += 1
    print(win_count, game_count)
    print(f"Games lost: {games_lost}")
    for games in games_lost:
        print("Player's Board:")
        games[0].display()
        print("Computer's Board:")
        games[1].display()

def main():
    stats()

if __name__ == "__main__":
    main()