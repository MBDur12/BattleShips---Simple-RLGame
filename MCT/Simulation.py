import numpy as np
import sys
sys.path.append("./Battleships")
import Board

def randomly_guess_ship(state, ship_length):
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
        guess_val = 0
        for coord in guessed_coords:
            if state.board[coord[0], coord[1]] == -1:
                break
            elif state.board[coord[0], coord[1]] == 0:
                contains_undiscovered = True
            else:
                guess_val += 1


        if contains_undiscovered:
            valid = True
        
        return guessed_coords, guess_val

def simulation(state, guess_limit):
    information_set = []
    for _ in range(guess_limit):
        # generate random ship and get its length
        index = np.random.randint(0, len(state._ships))
        ship_name = state._ships.keys[index]
        ship_length = state._ship_lengths[ship_name]

        # generate a guess for ship
        guessed_coords, guess_val = randomly_guess_ship(state, ship_length)
        print(guessed_coords, guess_val)

        if guess_val > 0:
            information_set


    


b = Board.Board(5)
print(randomly_place_ship(b, 3))