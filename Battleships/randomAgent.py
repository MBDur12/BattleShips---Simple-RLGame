import Board
import time
from random import randint
import numpy as np
import matplotlib.pyplot as plt

# How do I make it so that it prints in place?

def main():
    number_of_episodes = 1000
    total_moves_used = []
    for episode in range(1, number_of_episodes+1):
        num_moves = 0
        print(f"Episode #{episode}")
        board = Board.Board(10)
        board.place_ship("destroyer", 3)
        board.place_ship("a", 3)
        board.place_ship("b", 2)
        board.place_ship("c", 4)
        board.place_ship("d", 5)
        while board._ships:
            guess = tuple([randint(0, board.width-1), randint(0, board.width-1)])
            while board.board[guess[0], guess[1]] != 0:
                guess = tuple([randint(0, board.width-1), randint(0, board.width-1)])
            board.hit(guess)
            num_moves += 1
        
        total_moves_used.append(num_moves)
       
    
    print(f"Avg moves per game: {sum(total_moves_used)/len(total_moves_used)}")
    print(f"Min moves {min(total_moves_used)}. Max moves {max(total_moves_used)}")
    moves = np.array(total_moves_used)
    plt.hist(moves)
    plt.show()


if __name__ == "__main__":
    main()
