import numpy as np
from random import randint

class Board():
    def __init__(self, width):
        self.width = width
        self.board = np.zeros((width, width), dtype=int)
        # Store all positions occupied by ships
        self.ships = {}
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

    # Check if a given position is not already occuped and is a valid placement (i.e., within bounds)
    def _valid_position(self, pos):
        for coords in self.ships.values():
            if pos in coords:
                return False
        
        for val in pos:
            if val < 0 or val >= self.width:
                return False

        return True
        
    def place_ship(self, length, name):
        if length <= 0:
            print("Invalid Length")
            return

        x = randint(0, self.width)
        y = randint(0, self.width)

        while not self._valid_position([x,y]):
            x = randint(0, self.width)
            y = randint(0, self.width)

        self.ships[name] = [[x,y]]
        x_or_y = randint(0,1)
        for i in range(1, length):
            if x_or_y == 0:
                if self._valid_position([x-1,y]):
                    self.ships[name].append([x-1,y])
                else:
                    self.ships[name].append([x+1,y])
            if x_or_y == 1:
                if self._valid_position([x,y-1]):
                    self.ships[name].append([x,y-1])
                else:
                    self.ships[name].append([x,y+1])
    
    # Changes values in self.board depending on if a position is hit or not - reveals information.
    def hit(self, pos):
        for k,v in self.ships.items():
            if pos in v:
                v.remove(pos)
                self.board[pos[0], pos[1]] = 1
                print("Hit!") 
                if not v:
                    self.ships.pop(k, None)
                    print("Ship Sunk!")
                    return "Sunk"

                return "Hit"
        
        print("Miss!")
        self.board[pos[0], pos[1]] = -1
        return "Miss"

        


        



b = Board(5)

b.ships["a"] = [[0,1], [1,1]]
print(b.ships)
b.hit([0,1])
b.hit([1,1])
print(b.board)
