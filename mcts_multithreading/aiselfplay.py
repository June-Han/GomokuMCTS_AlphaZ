import numpy as np
import os

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
        self.ai1_wins = 0
        self.ai2_wins = 0
        self.game_results = None

        for i in range (0, self.total_games):
            self._selfplay()
            if self.game_results == 1:
                self.ai1_wins += 1
                print("Winner: ", self.result_text[PLAYER_A])
            elif self.game_results == -1:
                self.ai2_wins += 1
                print("Winner: ", self.result_text[PLAYER_B])
            self._print_details(i)
            self.game_board.reset_board()
            self.ai1.game_board.reset_board()
            self.ai1.reset_ai(GomokuBoard)
            self.ai2.game_board.reset_board()
            self.ai2.reset_ai(GomokuBoard)
            self.axes = None
        
        with open(os.path.join(os.getcwd(), "results.txt"), "a") as f:
            print(f"{os.getcwd()}{os.sep}results.txt")
            f.write("\nAI Player 1 (Black Stone) wins: " + str(self.ai1_wins))
            f.write("\nAI Player 2 (White Stone) wins: " + str(self.ai2_wins))
        

    def _selfplay(self):
        while self.game_board.judge() is None:
            #AI1:
            ai1_best_next_move = self.ai1.best_move()
            self.ai1.update_state(ai1_best_next_move)
            self.ai2.update_state(ai1_best_next_move)
            self.game_board.update_state(ai1_best_next_move)
            self.axes = self.game_board.draw(self.axes, 3)

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
            [game_no+1, "AI1 (Black Stone)", self.ai1.movecount, self.ai1.movetime, self.avg_time_ai1, self.result_text[self.game_results], self.ai1.min_num_sim],
            [game_no+1, "AI2 (White Stone)", self.ai2.movecount, self.ai2.movetime, self.avg_time_ai2, self.result_text[self.game_results],self.ai2.min_num_sim]
        ]
        table = tabulate(
            data, 
            headers=["Game No","AI_Player","Total No of Moves", "Total Time Taken", "Avg Time Per Move(s)", "Winner", "Total Simulations"], 
            tablefmt="grid"
        )

        with open(os.path.join(os.getcwd(), "results.txt"), "a") as f:
            f.write("\n")
            f.write(table)
            f.write("\n")

        print("Game ", game_no+1)
        print(table)



if __name__ == '__main__':
    gboard = GomokuBoard()
    ai1 = MCUCT(GomokuBoard, min_num_sim=2e4)
    ai2 = MCUCT(GomokuBoard, min_num_sim=3e4)
    app = ai_battle(gboard, ai1, ai2, 4)