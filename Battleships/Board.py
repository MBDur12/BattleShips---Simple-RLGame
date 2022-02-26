import numpy as np

class Board():
    def __init__(self, width):
        self.width = width
        self.board = np.zeros((width, width), dtype=int)
    def display(self):
        for row in self.board:
            print("-" * (4 * self.width + 1))
            print("| ", end="")
            for col in row:
                if col == 0:
                    print(" ", end="")
                else:
                    print("X", end="")
                print(" | ", end="")
            print("")
        print("-" * (4 * self.width + 1))


b = Board(5)
b.display()
