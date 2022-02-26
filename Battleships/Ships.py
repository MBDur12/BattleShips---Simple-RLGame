from random import randint

class Ship():
    def init(self, length):
        self.length = length
        self.position = []

    def _is_sunk(self):
        if self.position:
            return True

        return False 

    # If a matched position is hit, remove it from the remaining positions of the ship
    def hit(self, position):
        self.position.remove(position)
        if self._is_sunk:
            return "Sunk"
        else:
            return "Hit"

    def _available_pos(self,)

    # Generate a random board position for the ship.
    def set_position(self, board_size):
        x = randint(0, board_size-1)
        y = randint(0, board_size-1)
        self.position.append([x,y])
        # Generate an array of possible coordinates around the anchor
        # Dealing with ships overlapping








    
    