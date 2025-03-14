# Standard imports
import sys
import os
import getopt
from datetime import datetime as dt
import logging
import pickle
# Module imports
import plotly.graph_objects as go
# Game imports
from snake_api import SnakeApi
# AI imports
from ai_agent import Agent, LinearQNet, DeepQNet


def save_data(inference, out_dir, architecture_name, score):
    if inference:
        data_file = os.path.join(out_dir, architecture_name + '_inference.pkl')
    else:
        data_file = os.path.join(out_dir, architecture_name + '_train.pkl')
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            all_scores = pickle.load(f)
    else:
        all_scores = []
    all_scores.extend(score)
    with open(data_file, 'wb') as f:
        pickle.dump(all_scores, f)
    return all_scores


if __name__ == '__main__':
    DATE = str(dt.now().year) + str(dt.now().month) + str(dt.now().day) + '_' + str(dt.now().hour) + str(dt.now().minute) + str(dt.now().second)

    inference = True
    nb_game = 1
    weight_file = 'model_weights.pth'

    out_dir = DATE
    os.mkdir(out_dir)

    logging.basicConfig(filename=os.path.join(out_dir, 'log_' + DATE + '.log'), encoding='utf-8', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"htin:w:",["help", "train", "inference", "nb=", "weight="])
    except getopt.GetoptError:
        print('main.py --train --inference --nb <N> --weight <file.pth>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', "--help"):
            print('main.py --train --inference --nb <N> --weight <file.pth>')
            sys.exit()
        elif opt in ("-t", "--train"):
            inference = False
        elif opt in ("-i", "--inference"):
            inference = True
        elif opt in ("-n", "--nb"):
            nb_game = int(arg)
        elif opt in ("-w", "--weight"):
            weight_file = int(arg)

    game = SnakeApi()
    game.rewards = {
                    'collision': -10,
                    'self collision': -10,
                    'food': 10,
                    'closer_to_food': 2
                }
    
    agent = Agent(game, inference, nb_game, 1024, LinearQNet, weight_file)
    try:
        agent.train(10000)
    except Exception as e:
        logging.error(e)
        sys.exit(1)

    # Save the scores
    all_scores = save_data(inference, out_dir, 'data', agent.scores)
    
    # Display score graph
    x = list(range(len(all_scores)))
    fig = go.Figure(data=go.Scatter(x=x, y=all_scores, mode='lines+markers'))
    fig.write_html(os.path.join(out_dir, '_score.html'))
    fig.show()