# Standard imports
import sys
import getopt
from datetime import datetime as dt
import logging
# Game imports
from snake_api import SnakeApi
# AI imports
from ai_agent import Agent

DATE = str(dt.now().year) + str(dt.now().month) + str(dt.now().day) +  str(dt.now().hour) + str(dt.now().minute) + str(dt.now().second)


if __name__ == '__main__':
    logging.basicConfig(filename='log_' + DATE + '.log', encoding='utf-8', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())
    
    inference = True
    nb_game = 1

    try:
        opts, args = getopt.getopt(sys.argv[1:],"htin:",["help", "train", "inference", "nb="])
    except getopt.GetoptError:
        print ('main.py -i <inputfile> -o <outputfile>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', "--help"):
            print ('test.py --train <nbr game>')
            sys.exit()
        elif opt in ("-t", "--train"):
            inference = False
        elif opt in ("-i", "--inference"):
            inference = True
        elif opt in ("-n", "--nb"):
            nb_game = int(arg)

    game = SnakeApi()
    agent = Agent(game, inference, nb_game)
    agent.train()