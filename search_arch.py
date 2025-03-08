# Standard imports
import os
from datetime import datetime as dt
import logging
import json
import pickle
# Module imports
import plotly.graph_objects as go
# Game imports
from snake_api import SnakeApi
# AI imports
from ai_agent import Agent, LinearQNet, DeepQNet

from main import save_data


if __name__ == '__main__':
    DATE = str(dt.now().year) + str(dt.now().month) + str(dt.now().day) + '_' + str(dt.now().hour) + str(dt.now().minute) + str(dt.now().second)

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
    game_speed = 10000

    out_dir = DATE
    os.mkdir(out_dir)

    logging.basicConfig(filename=os.path.join(out_dir, 'log_' + DATE + '.log'), encoding='utf-8', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())  

    scores_by_model_architecture = {}

    game = SnakeApi()
    
    with open(os.path.join(out_dir, 'rewards.json'), 'w') as f:
        json.dump(rewards, f, indent=2)

    for network_type in [LinearQNet, DeepQNet]:
        for hidden_layer_size in [64, 128, 256, 512, 1024]:
            for n, reward in enumerate(rewards):
                game.rewards = reward
                # Train architecture
                architecture_name = network_type.__name__ + '_' + str(hidden_layer_size) + '_' + str(n)
                logging.info('Training ' + architecture_name)
                agent = Agent(game, inference, nb_game, hidden_layer_size, network_type, os.path.join(out_dir, architecture_name + '.pth'))
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