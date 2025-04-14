from board import Board
from board import PLAYER_A, PLAYER_B, TIE


class GomokuBoard(Board):

    search_directions = (
        (0, 1), (1, 0), (1, 1), (1, -1),
    )

    def __init__(self):
        Board.__init__(self)
        self.game_result = None

    def reset_board(self):
        super().reset_board()
        self.game_result = None

    def copy(self):
        new_board = Board.copy(self)
        Board.copy(new_board)
        setattr(new_board, 'game_result', self.game_result)
        return new_board

    def update_state(self, move):
        Board.update_state(self, move)
        self.game_result = self._judge(move)
        if self.game_result is not None:
            self._empty_indices = []

    def judge(self):
        return self.game_result

    def _judge(self, move):
        #Only need to search if the last move forms a five in a line
        #Return the player that wins if exists, otherwise return None
        if self.num_stones == Board.num_rows*Board.num_cols:
            return TIE
        last_color = self.last_player
        for dir in GomokuBoard.search_directions:
            max_stone_in_line = 1
            for sign in [-1, 1]:
                for offset in range(1, 5):
                    i = move[0] + dir[0]*offset*sign
                    j = move[1] + dir[1]*offset*sign
                    if 0 <= i < Board.num_rows and 0 <= j < Board.num_cols and self[i, j] == last_color:
                        max_stone_in_line += 1
                    else:
                        break
            if max_stone_in_line >= 5:
                self.judge = lambda : self.last_player
                return self.last_player
        return None