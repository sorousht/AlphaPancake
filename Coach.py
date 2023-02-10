import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS

log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.finished = 0 
        self.loopcount = 0 
    def executeEpisode(self,board):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.
        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.
        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        #board = self.game.getInitBoard()
        print(board)
        self.curPlayer = 1
        episodeStep = 0
        loop = {}
        stack = board.copy()
        while True:
            loop[self.game.stringRepresentation(stack.copy())]=1
            episodeStep += 1
            temp = int(episodeStep < self.args.tempThreshold)
            #pi = self.mcts.getActionProb(board, temp=temp)
            pi,Q  = self.mcts.getActionProb(stack, temp=temp)
            #v = sum([e*q for e in pi and q])
            trainExamples.append([stack,  pi,episodeStep])
            action = np.random.choice(len(pi), p=pi)
            #trainExamples.append([board,pi,Q[self.game.stringRepresentation(stack),action]])
            stack= self.game.getNextState(stack, action)
            r = self.game.getGameEnded(stack)
            '''
            if(loop.get(self.game.stringRepresentation(board))==1):
                action = np.random.choice(self.game.getActionSize())
                board= self.game.getNextState(board, action)
                if self.game.getGameEnded(board) ==1:
                    return [(x[0],x[1], r) for x in trainExamples]
            '''
            
            if r == 1:
                self.finished+=1
                return [(x[0],x[1], 1-x[2]) for x in trainExamples],True
                
            if episodeStep >9:
                return [(x[0],x[1], -x[2]) for x in trainExamples], False
            '''
            if r ==1:
                self.finished+=1
                return trainExamples
                print(stack)
                #return [(x[0],x[1], Q[(s,action)]) for x in trainExamples]
        
            if episodeStep >= 18:
                return None #trainExamples[0:len(trainExamples)-1]
                #eturn [(x[0],x[1], -1) for x in trainExamples]
            '''
    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            #board = self.game.getInitBoard()
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamplesP = deque([], maxlen=self.args.maxlenOfQueue)
                iterationTrainExamplesN = deque([], maxlen=self.args.maxlenOfQueue)
                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    board = self.game.getInitBoard()
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    e,t =self.executeEpisode(board) 
                    if t:
                        iterationTrainExamplesP += e
                    else:
                        iterationTrainExamplesN += e

                     
                # save the iteration examples to the history 
                #iterationTrainExamplesP *=int(len(iterationTrainExamplesN)/len(iterationTrainExamplesP))
                #self.trainExamplesHistory.append(iterationTrainExamplesP+iterationTrainExamplesN)
                self.trainExamplesHistory.append(iterationTrainExamplesP+iterationTrainExamplesN)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)
            print("loop",self.loopcount,"finished",self.finished)
            self.finished=0
            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)
            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(self.args, lambda x: np.argmax(pmcts.getActionProb(x, temp=0)[0]),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)[0]), self.game)
            pwins, nwins, draws,args = arena.playGames(self.args.arenaCompare)
            
            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                self.args = args

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True