import numpy as np
import torch

# Might remove this reproducible results later
torch.manual_seed(0)

from TreeNodes import Node


class MCTSParallel:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad() #Make the predictions and don't utilize the gradients for training.
    def search(self, states, SelfPlayGames):
        # Implementing policy from model 
        # (With a list of self-played games, there is a batch hence unsqueeze(0) is not required)
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
        )

        # Add Noise for more exploration during search: P(s,a) = (1 - epsilon)pa + epsilon(noise_a)
        # Add an extra size = policy.shape[0] for noise for the batch
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size= policy.shape[0])
        
        for i, spg in enumerate(SelfPlayGames):
            spg_policy = policy[i]
            valid_moves = self.game.get_valid_moves(states[i])
            spg_policy *= valid_moves #Mask out illegal moves
            spg_policy /= np.sum(spg_policy)

            spg.root = Node(self.game, self.args, states[i], visit_count=1)
            spg.root.expansion_with_policy(spg_policy)

        for search in range(self.args['num_searches']):
            # backpropagation #Return visit counts at the end
            for spg in SelfPlayGames:
                spg.node = None
                node = spg.root

                # selection
                while node.is_fully_expanded():
                    node = node.selection()

                #Is the selected node terminal else do expansion
                value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
                value = self.game.get_opponent_value(value)

                if is_terminal:
                    #mcts backpropagation
                    node.backpropagation(value)
                else:
                    spg.node = node
            
            # Store mapping Idx for each self play game
            expandable_SelfPlayGames = [mappingIdx for mappingIdx in range(len(SelfPlayGames)) if SelfPlayGames[mappingIdx].node is not None]

            if len(expandable_SelfPlayGames) > 0:
                states = np.stack([SelfPlayGames[mappingIdx].node.state for mappingIdx in expandable_SelfPlayGames])
                #Implementing the model============================================================================
                #Get encoded version of current state, convert it to tensor and set to unsqueeze the axes
                #Initialize an empty batch with just this one state.
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                )
                
                #Softmax on the 15*15 neurons, 1 per possible move.
                #Convert the logics(15*15 floats) into distribution of likelihoods
                policy = torch.softmax(policy, axis=1).cpu().numpy() #Use CPU 

            for i, mappingIdx in enumerate(expandable_SelfPlayGames):
                node = SelfPlayGames[mappingIdx].node
                spg_policy, spg_value = policy[i], value[i]

                #Mask the policy to not expand in direction a player has already played
                #Retrieve Valid Moves
                valid_moves = self.game.get_valid_moves(node.state)
                spg_policy *= valid_moves #All illegal moves will have policy 0

                #Rescale policies to retrieve percentages
                spg_policy /= np.sum(spg_policy) #So that sum of policy returned to 1
                #expansion mcts
                node.expansion_with_policy(spg_policy)  