# Standard imports
import os
from datetime import datetime as dt
import logging
import pickle
# Module imports
import plotly.graph_objects as go
# Game imports
from snake_api import SnakeApi
# AI imports
from ai_agent import Agent, LinearQNet, DeepQNet

DATE = str(dt.now().year) + str(dt.now().month) + str(dt.now().day) +  str(dt.now().hour) + str(dt.now().minute) + str(dt.now().second)


if __name__ == '__main__':
    out_dir = DATE
    os.mkdir(out_dir)

    logging.basicConfig(filename=os.path.join(out_dir, 'log_' + DATE + '.log'), encoding='utf-8', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())
    
    inference = False
    nb_game = 10
    game_speed = 10000

    scores_by_model_architecture = {}

    game = SnakeApi()

    for network_type in [LinearQNet, DeepQNet]:
        for hidden_layer_size in [64, 128, 256, 512, 1024]:
            # Train architecture
            architecture_name = network_type.__name__ + '_' + str(hidden_layer_size)
            logging.info('Training ' + architecture_name)
            agent = Agent(game, inference, nb_game, hidden_layer_size, network_type, os.path.join(out_dir, architecture_name + '.pth'))
            agent.train(game_speed)
            # Save the scores
            if inference:
                data_file = os.path.join(out_dir, architecture_name + '_inference.pkl')
            else:
                data_file = os.path.join(out_dir, architecture_name + '_train.pkl')
            if os.path.exists(data_file):
                with open(data_file, 'rb') as f:
                    all_scores = pickle.load(f)
            else:
                all_scores = []
            all_scores.extend(agent.scores)
            with open(data_file, 'wb') as f:
                pickle.dump(all_scores, f)
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