import numpy as np
import sys
sys.path.append("./Battleships")
import Board

def randomly_place_ship(state, ship_length):
    # Generate random, contiguous coordinates for the given ship length.
    valid = False
    while not valid:
        guessed_coords = []

        orientation = np.random.randint(0, 2) # 0 =x, 1= y(parts of the coordinates that don't change)
        c1 = c2 = 0
        
        # check to see if coordinates are sufficiently apart to "fill in" the rest: 
        # e.g., ship length = 3, then 2 and 4 with 3 in between for the coordinates.
        while abs(c1 - c2) != ship_length - 1:
            axis_val, c1, c2 = np.random.randint(0, state.width), np.random.randint(0, state.width), np.random.randint(0, state.width) # state should be passed in Board() instance.
        
        if c2 < c1: # switch coordinate 2 to be the upper bound
            c1, c2 = c2, c1

        if orientation == 0:
            for num in range(c1, c2+1):
                guessed_coords.append((axis_val, num))
        else:
            for num in range(c1, c2+1):
                guessed_coords.append((num, axis_val))

        # Next check conditions of validiity: if a "miss" is included and if there is at least one undiscovered square.
        contains_undiscovered = False
        for coord in guessed_coords:
            if state.board[coord[0], coord[1]] == -1:
                break
            if state.board[coord[0], coord[1]] == 0:
                contains_undiscovered = True

        if contains_undiscovered:
            valid = True
        
        
        return guessed_coords


    


b = Board.Board(5)
print(randomly_place_ship(b, 3))