import Board

def main():
    # initialise the game objects
    board = Board.Board(5)
    board.place_ship("carrier", 2)
    board.place_ship("destroyer", 3)
    board.display()
    # main game loop
    while board._ships:
        print(board._ships)
        # display board
        
        # input guess
        guess = input("Enter a coordinate (e.g. 0 1): ").split(" ")
        guess = tuple(map(lambda x: int(x), guess))
        print(guess)
        board.hit(guess)
        board.display()

if __name__ == '__main__':
    main()
