from Gomoku import Gomoku
from ResNet import ResNet
from AlphaZero import AlphaZero
from AlphaZeroParallel import AlphaZeroParallel

import torch

gomoku = Gomoku()

#Option for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Initialize model
model = ResNet(gomoku, 12, 256, device)

#Adam optimizer (weight decay due to 2 regularizations in loss for alphazero
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001 )

args = {
    'C': 2,
    'num_searches': 5000,
    'num_iterations': 8,
    #No. of selfplays for each iteration
    'num_selfPlay_iterations': 500,
    #No of games played in parallel
    'num_parallel_games': 100,
    'num_epochs': 4,
    'batch_size': 256,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}

alphaZero = AlphaZeroParallel(model, optimizer, gomoku, args)
alphaZero.learn()