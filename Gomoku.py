import numpy as np

class Gomoku:
    def __init__(self):
        self.row_count = 15
        self.column_count = 15
        self.match_row_count = 5
        self.match_col_count = 5
        self.action_size = self.row_count * self.column_count

    #Initialize entire board of zeros
    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))
    
    def get_next_state(self, state, action, player):
        #Ensures that moves stay within the grid, appropriate number for the stats\es.
        row = action // self.column_count
        column = action % self.column_count
        state[row, column] = player
        return state

    #Flatten out the states, and check if state is 0 (unsigned integers)
    # Gets true or false for entire board
    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)

    #Checking if the player have won.
    def check_win(self, state, action):
        if action == None:
            return False
        row = action // self.column_count
        column = action % self.column_count
        player = state[row, column]
        
        #Sum of state of the given row and all columns in the row, check if there's a matching 5 columns in 1 row
        search_directions = (
            (0, 1), (1, 0), (1, 1), (1, -1),
        )
        for dir in search_directions:
            max_stone_in_line = 1
            for sign in [-1, 1]:
                for offset in range(1, 5):
                    i = row + dir[0]*offset*sign
                    j = column + dir[1]*offset*sign
                    if 0 <= i < self.row_count and 0 <= j < self.column_count and state[i, j] == np.sign(player):
                        max_stone_in_line += 1
                    else:
                        break
            if max_stone_in_line >= self.match_row_count:
                return True
        return False

    #Check for draw and game termination
    def get_value_and_terminated(self, state, action):
        #Check if the game have been won.
        if self.check_win(state, action):
            return 1, True
        #Check for the draw
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        #In all other cases of game termination, return state 0 and action False
        return 0, False
        
    #Ensure distinction between 2 players
    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    #Change positive one to negative onces and vice versa
    def change_perspective(self, state, player):
        return state * player

    # Encode state to pass to model
    # 3 planes to pass to the model (state of all -1 or all 0s or all 1s)
    # state == -1 is the opponent, 1 is the player and 0 meaning empty.
    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state ==1)
        ).astype(np.float32)
        return encoded_state