from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
import numpy as np
import random
import torch
class PancakeGame(Game):
    square_content = {
        -1: "X",
        +0: "-",
        +1: "O"
    }
    

   
    def flip(self,lst, k):
        lst[:k+1] = lst[:k+1][::-1]
        return lst 


    def __init__(self, n):
        self.n = n
        self.sorted= [x for x in range(1,self.n+1)]

    def getInitBoard(self):
        # return initial board (numpy board)
        stack = [x for x in range(1,self.n+1)]
        while stack == [x for x in range(1,self.n+1)]:
            stack = random.sample(range(1, self.n+1), self.n)
        return np.array(stack)

    def stringRepresentation(self, stack):
        return  ''.join(map(str, list(stack)))


    def getActionSize(self):
        # return number of actions
        return self.n

    def getNextState(self, stack,action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        return self.flip(stack,action)

    def getValidMoves(self, board,l):
        # return a fixed size binary vector
        valids = valids = [0 if x == 0 or x==l else 1 for x in range(self.n)]
        return np.array(valids)

    def getGameEnded(self, stack):
        if list(stack) == self.sorted:
            return 1
        else:
            return 0

    def getTensor(self,stack): 
        positions= np.arange(self.n)
        stack= np.concatenate((stack.reshape(-1,1), positions.reshape(-1,1)), axis=1).astype(np.float64)
        stack= torch.FloatTensor(stack)
        stack = stack.view(1,self.n,2)
        return stack
    def getTensorFromBoards(self,stack): 
        l = len(stack)
        arr = np.arange(self.n)
        positions = np.tile(arr, l)
        stack= np.concatenate((stack.reshape(-1,1), positions.reshape(-1,1)), axis=1).astype(np.float64)
        stack= torch.FloatTensor(stack)
        stack = stack.view(l,self.n,2)
        return stack
    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l



    @staticmethod
    def display(board):
        print(''.join(map(str, list(board))))
    print("_")

