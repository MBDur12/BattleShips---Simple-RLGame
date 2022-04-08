import numpy as np
import sys
sys.path.append("./Battleships")
import Board

class BattleshipsMonteCarloTreeSearch():

    def __init__(self, state, guess_limit):
        self.state = state
        self.guess_limit = guess_limit

    def randomly_guess_ship(self, ship_length):
        # Generate random, contiguous coordinates for the given ship length.
        valid = False
        while not valid:
            guessed_coords = []

            orientation = np.random.randint(0, 2) # 0 =x, 1= y(parts of the coordinates that don't change)
            c1 = c2 = 0
            
            # check to see if coordinates are sufficiently apart to "fill in" the rest: 
            # e.g., ship length = 3, then 2 and 4 with 3 in between for the coordinates.
            while abs(c1 - c2) != ship_length - 1:
                axis_val, c1, c2 = np.random.randint(0, self.state.width), np.random.randint(0, self.state.width), np.random.randint(0, self.state.width) # state should be passed in Board() instance.
            
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
                if self.state.board[coord[0], coord[1]] == -1:
                    break
                elif self.state.board[coord[0], coord[1]] == 0:
                    contains_undiscovered = True
                else:
                    guess_val += 1


            if contains_undiscovered:
                valid = True
            
            return guessed_coords, guess_val

    def simulation(self, imperfect_board):
        prev_guesses = []
        for _ in range(self.guess_limit):
            # generate random ship and get its length
            index = np.random.randint(0, len(self.state._ships))
            ship_name = list(self.state._ships.keys())[index]
            ship_length = self.state._ship_lengths[ship_name]

            # generate a guess for ship
            guessed_coords, guess_val = self.randomly_guess_ship(ship_length)
            print(guessed_coords, guess_val)

            if guessed_coords in prev_guesses:
                continue
            else:
                prev_guesses.append(guessed_coords)

            if guess_val > 0:
                for coord in guessed_coords:
                    if self.state.board[coord[0], coord[1]] == 0:
                        imperfect_board[coord[0], coord[1]] += guess_val

        return imperfect_board

    # run simulations, and choose best action
    def best_action(self):
        # initialise imperfect board state
        imperfect_board = np.zeros((self.state.width, self.state.width))

        # loop board state, setting any explored coords to -1
        for index, val in np.ndenumerate(self.state.board):
            if val != 0:
                imperfect_board[index[0], index[1]] = -1
        
        # run the simulation
        imperfect_board = self.simulation(imperfect_board)
        print(imperfect_board)

        # get the best weighted action based on imperfect board
        best_action = np.unravel_index(imperfect_board.argmax(), imperfect_board.shape)
        print(best_action)

        return best_action

