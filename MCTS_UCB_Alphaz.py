import numpy as np
import torch

# Might remove this reproducible results later
torch.manual_seed(0)

from TreeNodes import Node


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad() #Make the predictions and don't utilize the gradients for training.
    def search(self, state):
        root = Node(self.game, self.args, state, visit_count=1)

        #Implementing policy from model
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )

        # Add Noise for more exploration during search: P(s,a) = (1 - epsilon)pa + epsilon(noise_a)
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        
        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves #Mask out illegal moves
        
        policy /= np.sum(policy)
        root.expansion_with_policy(policy)

        for search in range(self.args['num_searches']):
            # backpropagation #Return visit counts at the end
            node = root

            # selection
            while node.is_fully_expanded():
                node = node.selection()

            #Is the selected node terminal else do expansion
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)

            # expansion
            # Simulation/Playouts
            if not is_terminal:
                #Implementing the model============================================================================
                #Get encoded version of current state, convert it to tensor and set to unsqueeze the axes
                #Initialize an empty batch with just this one state.
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )
                
                #Softmax on the 15*15 neurons, 1 per possible move.
                #Convert the logics(15*15 floats) into distribution of likelihoods
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy() #Use CPU 
                
                #Mask the policy to not expand in direction a player has already played
                #Retrieve Valid Moves
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves #All illegal moves will have policy 0

                #Rescale policies to retrieve percentages
                policy /= np.sum(policy) #So that sum of policy returned to 1
                value = value.item()
                node.expansion_with_policy(policy)
                
                #MCTS=================================================================================================
                #node = node.expansion()
                #value = node.playout()

            #Backpropagation
            node.backpropagation(value)    
            
        #Which actions are promising, at the beginning it should be zero    
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs) #Convert to probability
        return action_probs