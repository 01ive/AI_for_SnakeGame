# Standard imports
import os
from datetime import datetime as dt
import logging
import json
import pickle
import getopt
import sys
# Module imports
import plotly.graph_objects as go
import torch

#Afficher ou non la fenetre du jeu
if "--headless" in sys.argv:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# Game imports
from snake_api import SnakeApi
# AI imports
from ai_agent import Agent, LinearQNet, DeepQNet

from main import save_data


if __name__ == '__main__':
    DATE = str(dt.now().year) + str(dt.now().month) + str(dt.now().day) + '_' + str(dt.now().hour) + str(dt.now().minute) + str(dt.now().second)

    criterions = [torch.nn.MSELoss(), torch.nn.SmoothL1Loss(), torch.nn.L1Loss(), torch.nn.HuberLoss(delta=0.5), torch.nn.HuberLoss(delta=10.0), torch.nn.HuberLoss(delta=30.0), torch.nn.HuberLoss(delta=50.0)]

    activations = [torch.nn.functional.leaky_relu, torch.nn.functional.gelu, torch.nn.functional.elu]

    rewards = [ {
                    'collision': -10,
                    'self collision': -10,
                    'food': 10,
                    'closer_to_food': 5 
                }, {
                    'collision': -10,
                    'self collision': -10,
                    'food': 10,
                    'closer_to_food': 2
                }, {
                    'collision': -20,
                    'self collision': -20,
                    'food': 50,
                    'closer_to_food': 5 
                } ]

    inference = False
    nb_game = 500
    game_speed = 1000000000

    out_dir = f"logs/{DATE}"
    os.mkdir(out_dir)

    logging.basicConfig(filename=os.path.join(out_dir, 'log_' + DATE + '.log'), encoding='utf-8', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())  

    try:
        opts, args = getopt.getopt(sys.argv[1:],"htin:w:",["help", "train", "inference", "nb=", "weight=", "headless"])
    except getopt.GetoptError:
        print('main.py --train --inference --nb <N> --weight <file.pth> --headless')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', "--help"):
            print('search_arch.py --headless')
            sys.exit()
        elif opt in ("--headless"):
            pass

    scores_by_model_architecture = {}

    game = SnakeApi()
    
    with open(os.path.join(out_dir, 'rewards.json'), 'w') as f:
        json.dump(rewards, f, indent=2)

    for network_type in [LinearQNet, DeepQNet]:
        for hidden_layer_size in [64, 128, 256, 512, 1024]:
            for criterion in criterions:
                for activation in activations:
                    for n, reward in enumerate(rewards):
                        game.rewards = reward
                        # Train architecture
                        architecture_name = network_type.__name__ + '_' + str(criterion).split('.')[-1][0:-2] + '_' + str(activation).split('.')[-1][1:-3].split()[1] + '_' + str(hidden_layer_size) + '_' + str(n)
                        logging.info('=========Training ' + architecture_name + '=========')
                        agent = Agent(game, criterion, activation, inference, nb_game, hidden_layer_size, network_type, os.path.join(out_dir, architecture_name + '.pth'))
                        agent.train(game_speed)
                        # Save the scores
                        all_scores = save_data(inference, out_dir, architecture_name, agent.scores)
                        # Save model architecture score
                        scores_by_model_architecture[architecture_name] = all_scores
    
    # Display score graph
    logging.info('Displaying score graph')
    x = list(range(nb_game))
    fig = go.Figure()
    for arch_name in scores_by_model_architecture.keys():
        fig.add_trace(go.Scatter(x=x, y=scores_by_model_architecture[arch_name], mode='lines+markers', name=arch_name))
    fig.write_html(os.path.join(out_dir, 'search_arch_score.html'))
    fig.show()