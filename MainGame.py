from Gomoku import Gomoku
import numpy as np
from MCTS_UCB import MCTS
from ResNet import ResNet
import torch

#Might remove this reproducible results later
torch.manual_seed(0)

gomoku = Gomoku()
player = 1

#C is sqrt 2 = 1.41, what is generally being used
#Removal of sqrt for the alphaZero
args = {
    'C' : 2,
    'num_searches': 10000,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}

#Implementation of the model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ResNet(gomoku, 12, 256, device)
# model.eval()

# mcts = MCTS(gomoku, args, model)
mcts = MCTS(gomoku, args)
state = gomoku.get_initial_state()

while True:
    print(state)

    if player == 1:
        valid_moves = gomoku.get_valid_moves(state)
        print("valid_moves", [i for i in range(gomoku.action_size) if valid_moves[i] == 1])
        action = int(input(f"{player}:"))
        print (action)
        
        if valid_moves[action] == 0:
            print("action not valid")
            continue
    else:
        neutral_state = gomoku.change_perspective(state, player)
        mcts_probs = mcts.search(neutral_state)
        action = np.argmax(mcts_probs)
        
    state = gomoku.get_next_state(state, action, player)
    
    value, is_terminal = gomoku.get_value_and_terminated(state, action)
    
    if is_terminal:
        print(state)
        if value == 1:
            print(player, "won")
        else:
            print("draw")
        break
        
    player = gomoku.get_opponent(player)
