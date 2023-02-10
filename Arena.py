import logging
import math
from tqdm import tqdm

log = logging.getLogger(__name__)


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self,args, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.
        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        self.maxD1 = 0
        self.maxD1 = 0
        self.args = args


    def playGame(self, verbose=False):
        """
        Executes one episode of a game.
        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player1, self.player2]
        curPlayer = 1
        board = self.game.getInitBoard()
        it_1 = 0
        it_2 = 0
        temp = board.copy()
        loop = {}
        loop[self.game.stringRepresentation(temp)] =1
        min =0
        while self.game.getGameEnded(temp) == 0:
            it_1 += 1
            if verbose:
                assert self.display
                print("Turn ", str(it_1), "Player ", str(curPlayer))
                self.display(temp)
            action = players[0](temp)
            '''
            valids = self.game.getValidMoves(temp,0)
             
            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            '''
            temp = self.game.getNextState(temp, action)
            if (it_1>=16):
                it_1 = math.inf
                #print("break 1")
                break
            '''
            if (loop.get(self.game.stringRepresentation(temp))==1):
                it = math.inf
                #print(self.game.stringRepresentation(temp),"loop")
                break
            else:
                loop[self.game.stringRepresentation(temp)]=1
            '''
        print("end player 1",self.game.stringRepresentation(temp))
        '''
        if (self.game.getGameEnded(temp)==0):
            if it_1 < self.args['minDepth']:
                self.args['minDepth'] =it_1
        '''
        temp = board.copy()
        loop = {}
        loop[self.game.stringRepresentation(temp)] =1
        curPlayer =2
        while self.game.getGameEnded(temp) == 0:
            it_2 += 1
            if verbose:
                assert self.display
                #print("Turn ", str(it_2), "Player ", str(curPlayer))
                self.display(temp)
            action = players[1](temp)
            print("action",action)
            if (it_2>=16):
                it_2 = math.inf
                #print("break 2")
                break
            '''
            valids = self.game.getValidMoves(temp,0)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            '''
            temp = self.game.getNextState(temp, action)
            '''
            if (loop.get(self.game.stringRepresentation(temp))==1):
                it_2 = math.inf
                #print(self.game.stringRepresentation(temp),"loop")
                break
            else:
                loop[self.game.stringRepresentation(temp)]=1
            '''
        print("end player 2",self.game.stringRepresentation(temp))
        if (self.game.getGameEnded(temp)==1):
            min = it_2
        if verbose:
            assert self.display
            #print("Game over: Turn ", str(it_2), "Result ", str(self.game.getGameEnded(board)))
            self.display(board)
        if it_2 < it_1:
            return 1,min
        elif it_2 == it_1:
            return 0,min
        else:
            return -1,min

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.
        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        oneWon = 0
        twoWon = 0
        draws = 0
        minD = []
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameResult,min = self.playGame(verbose=verbose)
            if gameResult == 1:
                twoWon += 1
            elif gameResult == -1:
                oneWon += 1
            else:
                draws += 1
            minD.append(min)
        print(max(minD))
        if twoWon >0:
            self.args['minDepth']=max(minD)
        return oneWon, twoWon, draws, self.args