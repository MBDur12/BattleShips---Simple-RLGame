import numpy as np
from random import randint
from itertools import product, groupby
from operator import itemgetter

class Board():
    def __init__(self, width):
        self.width = width
        self.board = np.zeros((width, width), dtype=int)
        # Store all positions occupied by ships
        self._ships = {}
    def display(self):
        for row in self.board:
            print("-" * (4 * self.width + 1))
            print("| ", end="")
            for col in row:
                if col == 0:
                    print(" ", end="")
                elif col == -1:
                    print("X", end="")
                else:
                    print("O", end="")
                print(" | ", end="")
            print("")
        print("-" * (4 * self.width + 1))

    def reset(self):
        self.board = np.zeros((self.width, self.width), dtype=int)
        self._ships = {}

    # Check if a given position is not already occuped and is a valid placement (i.e., within bounds)
    def _valid_position(self, pos):
        for coords in self._ships.values():
            if pos in coords:
                return False
        
        for val in pos:
            if val < 0 or val >= self.width:
                return False

        return True

    
        
    def generate_position(self, coords, ship_length, orientation, anchor): 
        if len(coords) < ship_length:
            return False

        pos = []

        if orientation == 0:
            line = [coord[1] for coord in coords if coord[0] == anchor]
        else:
            line = [coord[0] for coord in coords if coord[1] == anchor]
        
        cont_coords = []
        for key, group in groupby(enumerate(line), lambda x: x[0] - x[1]):
            cont_coords.append(list(map(itemgetter(1), group)))

        for set in cont_coords:
            if len(set) == ship_length:
                pos = set
                break
            elif len(set) > ship_length:
                pos = set[:ship_length]
                break

        if not pos:

            return False
        
        if orientation == 0:
            pos = [coord for coord in coords if coord[0] == anchor and coord[1] in pos]
        else:
            pos = [coord for coord in coords if coord[1] == anchor and coord[0] in pos]

   
        return pos
        
        
            
    def place_ship(self, ship_name, ship_length):
        
        # Generate all coordinates and then filter the positions (coords) that are currently unoccupied
        coords = list(product(range(self.width), repeat=2))
        spaces = [coord for coord in coords if self._valid_position(coord)]

        while True:
            anchor_val = randint(0,self.width-1)
            orientation = randint(0,1)
            
            set_pos = self.generate_position(spaces, ship_length, orientation, anchor_val)
            if set_pos:
                break
        
        self._ships[ship_name] = set_pos

       

    
    # Changes values in self.board depending on if a position is hit or not - reveals information.
    def _check_sunk(self, ship):
        if not self._ships[ship]:
            self._ships.pop(ship, None)
            return True
        return False

    def _check_hit(self, ship, coords, pos):
        if pos in coords:
            coords.remove(pos)
            self.board[pos[0], pos[1]] = 1
            return True
        

    def hit(self, pos):
        if self.board[pos[0], pos[1]] != 0:
            return
        else:
            for k,v in self._ships.items():
                if self._check_hit(k, v, pos):
                #print("Hit!")
            
                    if self._check_sunk(k):
                        print("Ship Sunk!", end="\r")
                    return
            self.board[pos[0], pos[1]] = -1





        
        #print("Miss!")
