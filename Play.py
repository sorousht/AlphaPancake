import Arena
from MCTS import MCTS
from PancakeGame import PancakeGame as Game
from Pancake.Pytorch.NNet import NNetWrapper as NNet
import os
import sys
import time
from utils import *
import numpy as np
from tqdm import tqdm


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

mini_othello = False  # Play in 6x6 instead of the normal 8x8.
human_vs_cpu = True

g = Game(5)

t1 = g.getInitBoard()


# nnet players
n1 = NNet(g)

n1.load_checkpoint('/content/drive/MyDrive/alphapancakeZ','temp.pth.tar')

args1 = dotdict({'numMCTSSims': 20, 'cpuct':1,'minDepth':20})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0)[0])


n2 = NNet(g)
n2.load_checkpoint('/content/drive/MyDrive/alphapancakeZ','best.pth.tar')
args2 = dotdict({'numMCTSSims': 20, 'cpuct': 1})
mcts2 = MCTS(g, n2, args1)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0)[0])

player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

arena = Arena.Arena(args1,n1p, player2, g, display=Game.display)

print(arena.playGames(40, verbose=True))