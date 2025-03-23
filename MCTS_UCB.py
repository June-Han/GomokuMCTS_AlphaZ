import numpy as np
import torch

print(np.__version__)
print(torch.__version__)
print(torch.version.cuda)

# Might remove this reproducible results later
torch.manual_seed(0)

from TreeNodes import Node


class MCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args

    def search(self, state):
        root = Node(self.game, self.args, state)

        for search in range(self.args['num_searches']):
            # backpropagation #Return visit counts at the end
            node = root

            # selection
            while node.is_fully_expanded_Non_AlphaZ():
                node = node.selection_Non_AlphaZ()

            #Is the selected node terminal else do expansion
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)

            # expansion
            # Simulation/Playouts
            if not is_terminal:
                node = node.expansion()
                value = node.playout()

            #Backpropagation
            node.backpropagation(value)    
            
        #Which actions are promising, at the beginning it should be zero    
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs) #Convert to probability
        return action_probs