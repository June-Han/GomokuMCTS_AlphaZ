import numpy as np
import math

#Function for each node
class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.visit_count = visit_count
        
        self.children = []
        #Expandable moves from each node
        self.expandable_moves = game.get_valid_moves(state) 
        self.value_sum = 0

        #Policy given to the action from parent perspective, utilizing the resnet layers
        self.prior = prior

    # Check if the depth is fully expanded from the node (children =/= 0)
    def is_fully_expanded(self):
        #np.sum(self.expandable_moves) == 0 and
        return len(self.children) > 0
    
    # Check if the depth is fully expanded from the node (children =/= 0)
    def is_fully_expanded_Non_AlphaZ(self):
        return np.sum(self.expandable_moves) == 0 and len(self.children) > 0

    #Selection of MCTS with UCB
    def selection(self):
        best_child = None
        best_ucb = -np.inf #negative infinity
        
        for child in self.children:
            ucb = self.get_ucb_Azero(child)
            # Update best ucb with new ucb retrieved from child.
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    #Selection of MCTS with UCB
    def selection_Non_AlphaZ(self):
        best_child = None
        best_ucb = -np.inf #negative infinity
        
        for child in self.children:
            ucb = self.get_ucb(child)
            # Update best ucb with new ucb retrieved from child.
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child

    # Q value - likelihood of winning for a given node (preferred in between 0-1 for probability)
    # C - choice for exploitation or exploration (constant)
    # n(s) - Visit count of parent
    # n(s,a) - Visit count of child
    # Q(s,a) + C*sqrt(ln(n(s))/n(s,a))
    # For the parent node, it should be max value, hence 1 - child value should be max (Standard MCTS)
    def get_ucb(self, child):
        q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2 #Between -1 and +1 hence add 1 then divide by 2.
        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)

    # U(s,a) = Q(s,a) + C*P(s,a)*sqrt(summationUpToB(N(s,b))/(1+N(s,a)))
    def get_ucb_Azero(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2 #Between -1 and +1 hence add 1 then divide by 2.
        return q_value + self.args['C'] * math.sqrt((self.visit_count) / (1 + child.visit_count)) * child.prior

    # Expansion part of MCTS
    def expansion(self):
        #random choice from the list of expandable moves (np.where to get indices from all available moves) & take the first one
        action = np.random.choice(np.where(self.expandable_moves == 1)[0])
        self.expandable_moves[action] = 0
        
        child_state = self.state.copy()
        child_state = self.game.get_next_state(child_state, action, 1)
        child_state = self.game.change_perspective(child_state, player=-1)
        
        child = Node(self.game, self.args, child_state, self, action)
        self.children.append(child)
        return child

    #Expansion part using policy from trained ResNet model
    def expansion_with_policy(self, policy):
        for action, probability in enumerate(policy):
            if probability > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)
        
                child = Node(self.game, self.args, child_state, self, action, probability)
                self.children.append(child)
                
        return child

    #Simulation and playout step
    def playout(self):
        value, is_terminal = self.game.get_value_and_terminated(self.state, self.action_taken)
        value = self.game.get_opponent_value(value)
        
        if is_terminal:
            return value
        
        playout_state = self.state.copy()
        playout_player = 1
        while True:
            valid_moves = self.game.get_valid_moves(playout_state)
            action = np.random.choice(np.where(valid_moves == 1)[0])
            playout_state = self.game.get_next_state(playout_state, action, playout_player)
            value, is_terminal = self.game.get_value_and_terminated(playout_state, action)
            if is_terminal:
                # Meaning it's opponent, and thus swap it to opponent action to consider from this node
                if playout_player == -1:
                    value = self.game.get_opponent_value(value)
                return value    
            #Flip player for next node in consideration, which is the opponent
            playout_player = self.game.get_opponent(playout_player)

    # Updating the values in the tree
    def backpropagation(self, value):
        self.value_sum += value
        self.visit_count += 1

        #The game takes turn to play, hence swap the node to opponent
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagation(value)  