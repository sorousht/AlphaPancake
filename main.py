import logging

import coloredlogs

from Coach import Coach
from PancakeGame import PancakeGame as Game
from Pancake.Pytorch.NNet import NNetWrapper as nn
log = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')  # Change this to DEBUG to see more info.
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
args = dotdict({
    'numIters': 100,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 40,        #
    'updateThreshold': 0.51,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 300,         # Number of games moves for MCTS to simulate.
    'arenaCompare': 120,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'minDepth':32,
    'checkpoint': '/content/drive/MyDrive/alphapancakeZ',
    'load_model': True,
    'load_folder_file': ('/content/drive/MyDrive/alphapancakeZ','best.pth.tar'),
    'numItersForTrainExamplesHistory': 10,

})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(5)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)
    
    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()