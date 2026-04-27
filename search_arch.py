# Standard imports
import os
import time
from datetime import datetime as dt
import logging
import json
import pickle
import getopt
import sys
# Module imports
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

#Afficher ou non la fenetre du jeu
if "--headless" in sys.argv:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# Game imports
from snake_api import SnakeApi
# AI imports
from ai_agent import Agent, LinearQNet, DeepQNet

from main import save_data


def format_time(seconds):
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)

    if h > 0:
        return f"{int(h)}h {int(m)}m {s:.2f}s"
    elif m > 0:
        return f"{int(m)}m {s:.2f}s"
    else:
        return f"{s:.2f}s"


if __name__ == '__main__':
    DATE = str(dt.now().year) + str(dt.now().month) + str(dt.now().day) + '_' + str(dt.now().hour) + str(dt.now().minute) + str(dt.now().second)

    criterions = [torch.nn.MSELoss(), torch.nn.SmoothL1Loss(), torch.nn.L1Loss()]#, torch.nn.HuberLoss(delta=0.5), torch.nn.HuberLoss(delta=10.0), torch.nn.HuberLoss(delta=30.0), torch.nn.HuberLoss(delta=50.0)]

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

    networks_types = [LinearQNet, DeepQNet]
    hidden_layers = [64, 128]#, 256, 512, 1024]

    inference = False
    nb_game = 25
    game_speed = 0

    out_dir = f"logs/{DATE}"
    os.makedirs(out_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(out_dir, 'log_' + DATE + '.log'), encoding='utf-8', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())

    is_headless = False
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
            is_headless = True

    scores_by_model_architecture = {}
    times_str_by_architecture = {}
    times_sec_by_architecture = {}

    hyperparam_results = {
        'Network': {},
        'Hidden Layer': {},
        'Criterion': {},
        'Activation': {},
        'Reward': {}
    }

    game = SnakeApi(headless = is_headless)
    
    with open(os.path.join(out_dir, 'rewards.json'), 'w') as f:
        json.dump(rewards, f, indent=2)

    t_ini = time.time()
    logging.info("Total iterations =  " + str(len(networks_types) * len(hidden_layers) * len(criterions) * len(activations) * len(rewards)))
    for network_type in networks_types:
        for hidden_layer_size in hidden_layers:
            for criterion in criterions:
                for activation in activations:
                    for n, reward in enumerate(rewards):
                        t1 = time.time()
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

                        max_score = max(all_scores)
                        params_current_run = [
                            ('Network', network_type.__name__),
                            ('Hidden Layer', str(hidden_layer_size)),
                            ('Criterion', str(criterion).split('.')[-1][0:-2]),
                            ('Activation', str(activation).split('.')[-1][1:-3].split()[1]),
                            ('Reward', f"Reward {n}")
                        ]

                        for category, param in params_current_run:
                            if param not in hyperparam_results[category]:
                                hyperparam_results[category][param] = []
                            hyperparam_results[category][param].append(max_score)

                        t2 = time.time()
                        execution_time = t2 - t1
                        times_sec_by_architecture[architecture_name] = execution_time
                        times_str_by_architecture[architecture_name] = format_time(execution_time)
                        print(f"Training time = {times_str_by_architecture[architecture_name]}")

    t_final = time.time()
    print("Total training time = " + format_time(t_final - t_ini))
    # Display score graph
    logging.info('Displaying score graph')
    x = list(range(nb_game))
    fig = go.Figure()
    fig.update_layout(title=f"Évolution des scores (Temps total : {format_time(t_final - t_ini)})", xaxis_title="Parties",
                       yaxis_title="Score")
    for arch_name in scores_by_model_architecture.keys():
        fig.add_trace(go.Scatter(x=x, y=scores_by_model_architecture[arch_name], mode='lines+markers', name=arch_name))
    fig.write_html(os.path.join(out_dir, 'search_arch_score.html'))
    fig.show()

    # ==========================================
    # GRAPHIQUE 2 : Bilan Score Max vs Temps (Double Axe Y)
    # ==========================================
    logging.info('Displaying summary graph (Score vs Time)')

    architectures = list(scores_by_model_architecture.keys())
    # On récupère le meilleur score de chaque architecture pour la comparaison
    max_scores = [max(scores_by_model_architecture[arch]) for arch in architectures]
    times_in_seconds = [times_sec_by_architecture[arch] for arch in architectures]

    # Création du graphique avec un axe Y secondaire
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])

    # Ajout de la courbe des Scores (Axe Y Primaire, à gauche)
    fig2.add_trace(
        go.Bar(x=architectures, y=max_scores, name="Score Max", marker_color='royalblue', opacity=0.7),
        secondary_y=False,
    )
    # Ajout de la courbe du Temps (Axe Y Secondaire, à droite)
    fig2.add_trace(
        go.Scatter(x=architectures, y=times_in_seconds, name="Temps (secondes)", mode='lines+markers',
                   marker_color='crimson', line=dict(width=3)),
        secondary_y=True,
    )

    # Mise en forme du graphique de synthèse
    fig2.update_layout(
        title_text="Bilan des Architectures : Score Maximum vs Temps d'entraînement",
        xaxis_title="Architectures",
        barmode='group'
    )
    fig2.update_yaxes(title_text="<b>Score Max</b>", secondary_y=False, color="royalblue")
    fig2.update_yaxes(title_text="<b>Temps d'entraînement (s)</b>", secondary_y=True, color="crimson")

    fig2.write_html(os.path.join(out_dir, 'summary_score_vs_time.html'))
    fig2.show()

    # ==========================================
    # GRAPHIQUE 3 : Moyenne des scores maxs par hyperparamètre
    # ==========================================
    logging.info('Displaying average performance by hyperparameter')

    fig3 = make_subplots(rows=1, cols=5, subplot_titles=list(hyperparam_results.keys()))

    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

    for i, (category, values_dict) in enumerate(hyperparam_results.items(), start=1):
        avg_scores = {k: (sum(v) / len(v)) for k, v in values_dict.items()}

        sorted_avg = dict(sorted(avg_scores.items(), key=lambda item: item[1], reverse=True))

        fig3.add_trace(
            go.Bar(
                x=list(sorted_avg.keys()),
                y=list(sorted_avg.values()),
                name=category,
                marker_color=colors[i - 1],
                text=[f"{val:.1f}" for val in sorted_avg.values()],  # Affiche la valeur sur la barre
                textposition='auto',
                showlegend=False
            ),
            row=1, col=i
        )

    fig3.update_layout(
        title_text="Score Max Moyen isolé par Hyperparamètre",
        yaxis_title="Score Moyen",
        height=500,
        template="plotly_white"
    )

    fig3.write_html(os.path.join(out_dir, 'average_hyperparameters.html'))
    fig3.show()
