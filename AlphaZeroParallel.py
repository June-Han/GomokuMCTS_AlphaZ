import numpy as np
import torch

print(np.__version__)
print(torch.__version__)
print(torch.version.cuda)

#Might remove this reproducible results later
torch.manual_seed(0)

import torch.nn as nn
import torch.nn.functional as nnFunctional
import random
from tqdm import tqdm
from SelfPlayGameClass import SelfPlayGame

from MCTS_UCB_Alphaz_Parallel import MCTSParallel

class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)
        
    def selfPlay(self):
        return_memory = []
        player = 1
        SelfPlayGamesList = [SelfPlayGame(self.game) for spg in range(self.args['num_parallel_games'])]
        
        while len(SelfPlayGamesList) > 0:
            states = np.stack([spg.state for spg in SelfPlayGamesList])
            current_states = self.game.change_perspective(states, player)
            self.mcts.search(current_states, SelfPlayGamesList)

            #Remove Terminal ones and the len will differ, so the number is flipped to cater to this.
            #Loop in opposite direction
            for i in range(len(SelfPlayGamesList))[::-1]:
                spg = SelfPlayGamesList[i]

                #Which actions are promising, at the beginning it should be zero    
                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= np.sum(action_probs) #Convert to probability
            
                spg.memory.append((spg.root.state, action_probs, player))

                #Give flexibility for exploit or explore
                #Higher the temperature, the smaller the power in the calculation, more exploration as probabilities gets squished together
                #Smaller Temperature meaning, higher exploitation
                temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                action = np.random.choice(self.game.action_size, p = action_probs)
                
                spg.state = self.game.get_next_state(spg.state, action, player)
                
                value, is_terminal = self.game.get_value_and_terminated(spg.state, action)
                
                if is_terminal:
                    for hist_current_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                        return_memory.append((
                            self.game.get_encoded_state(hist_current_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    del SelfPlayGamesList[i]
            
            player = self.game.get_opponent(player)
        return return_memory
                
            
    #Loop over memory in batches
    #From each batch and batch index, sample a whole batch of different samples and use these for training.
    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            #Ensure did not exceed memory
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])] # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
            state, policy_targets, value_targets = zip(*sample) #Transpose sample into list of 3s
            
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)
            
            out_policy, out_value = self.model(state)
            
            policy_loss = nnFunctional.cross_entropy(out_policy, policy_targets) #Multitarget cross-entropy loss
            value_loss = nnFunctional.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []
            
            self.model.eval()
            for selfPlay_iteration in tqdm(range(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games'])):
                memory += self.selfPlay()
                
            self.model.train()
            #tqdm for visualizing bars.
            for epoch in tqdm(range(self.args['num_epochs'])):
                self.train(memory)
            
            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")