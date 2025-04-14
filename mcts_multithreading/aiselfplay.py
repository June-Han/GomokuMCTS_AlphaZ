import numpy as np

from board import PLAYER_A, PLAYER_B, TIE
from gomoku_board import GomokuBoard
from MCUCT import MCUCT

# Import the tabulate module
from tabulate import tabulate

ax = None


#class ai_battle(tkinter.Canvas):
class ai_battle():
# Player A: Black Stone
# Player B: White Stone
# AI1: Black Stone
# AI2: White Stone
    result_text = {
        PLAYER_A: 'AI1 win',
        PLAYER_B: 'AI2 win',
        TIE: 'TIE',
    }

    def __init__(self, game_board, ai1, ai2, total_games):
        self.total_games = total_games
        self.game_board = game_board
        self.ai1 = ai1
        self.ai2 = ai2
        self.axes = None
        ai1_wins = 0
        ai2_wins = 0
        self.game_results = None

        for i in range (0, self.total_games):
            self._selfplay()
            if self.game_results == PLAYER_A:
                self.ai1_wins += 1
            elif self.game_results == PLAYER_B:
                self.ai2_wins += 1
            self._print_details(i)
            self.game_board.reset_board()
        
        f = open("results.txt", "a")
        f.write("AI Player 1 (Black Stone) wins:", self.ai1_wins)
        f.write("AI Player 2 (White Stone) wins:", self.ai2_wins)
        f.write("AI Player 2 (White Stone) avg time:", self.ai2_wins)
        f.close()
        

    def _selfplay(self):
        while self.game_board.judge() is None:
            #AI1:
            ai1_best_next_move = self.ai1.best_move()
            self.ai1.update_state(ai1_best_next_move)
            self.ai2.update_state(ai1_best_next_move)
            self.game_board.update_state(ai1_best_next_move)
            self.axes = self.game_board.draw(self.axes, 1)

            #AI2
            if self.game_board.judge() is None:
                ai2_best_next_move = self.ai2.best_move()
                self.ai2.update_state(ai2_best_next_move)
                self.ai1.update_state(ai2_best_next_move)
                self.game_board.update_state(ai2_best_next_move)
                self.axes = self.game_board.draw(self.axes, 3)
            self.game_results = self.game_board.judge()

    def _print_details(self, game_no=0):
        self.avg_time_ai1 = self.ai1.movetime/ai1.movecount
        self.avg_time_ai2 = self.ai2.movetime/ai2.movecount
        data = [
            ["AI1 (Black Stone)", self.ai1.movecount, self.ai1.movetime, self.avg_time_ai1, self.result_text[self.game_results], self.ai1.min_num_sum],
            ["AI2 (White Stone)", self.ai2.movecount, self.ai2.movetime, self.avg_time_ai2, self.result_text[self.game_results],self.ai2.min_num_sum]
        ]
        table = tabulate(
            data, 
            headers=["AI_Player","Total No of Moves", "Total Time Taken", "Avg Time Per Move(s)", "Winner", "Total Simulations"], 
            tablefmt="grid"
        )

        f = open("results.txt", "a")
        f.write(game_no + 1)
        f.write(table)
        f.close()

        print("Game ", game_no+1)
        print(table)



if __name__ == '__main__':
    gboard = GomokuBoard()
    ai1 = MCUCT(GomokuBoard, min_num_sim=2e4)
    ai2 = MCUCT(GomokuBoard, min_num_sim=3e4)
    app = ai_battle(gboard, ai1, ai2, 1)