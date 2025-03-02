import torch
import numpy as np
import json
from datetime import datetime as dt
import os
import sys
import getopt
import logging

# Game imports
from snake_api import SnakeApi, EndOfSnakeGame

# AI imports
from ai_agent import Agent

DATE = str(dt.now().year) + str(dt.now().month) + str(dt.now().day) +  str(dt.now().hour) + str(dt.now().minute) + str(dt.now().second)

def log_iteration(game_id, state, action, reward, done):
    log_data = {}
    log_file = "logs_"+ DATE + ".json"

    #Charger les logs existants s'il y en a
    try:
        with open(log_file, "r") as file:
            log_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        pass  # Fichier vide ou inexistant, on commence un nouveau log

    # Convertir les `numpy.ndarray` en listes pour éviter l'erreur JSON
    state = state.tolist() if isinstance(state, np.ndarray) else state
    action = action.tolist() if isinstance(action, np.ndarray) else action
    reward = float(reward) if isinstance(reward, np.ndarray) else reward
    done = bool(done) if isinstance(done, np.ndarray) else done

    # Ajouter la nouvelle entrée
    if str(game_id) not in log_data:
        log_data[str(game_id)] = []  # S'assurer que game_id est une clé de type str

    log_data[str(game_id)].append({
        "state": state,
        "action": action,
        "reward": reward,
        "done": done
    })

    # Sauvegarder le fichier mis à jour
    with open(log_file, "w") as file:
        json.dump(log_data, file, indent=4)

def end_of_iteration(agent, game, score, scores):
    # Réinitialisation du jeu et mise à jour des statistiques
    game.reset()
    agent.n_games += 1
    scores.append(score)
    logging.info('Game {} score {}'.format(agent.n_games, score))

def end_of_game(agent, scores):
    # save weight every 10 games
    logging.debug('Save weights to model_weights.pth')
    torch.save(agent.model.state_dict(), 'model_weights.pth')
    logging.info('Mean score: {}'.format(np.array(scores).mean()))
    logging.info('Var score: {}'.format(np.array(scores).var()))
    quit()

def train(nb_game, play_only=False):
    agent = Agent()
    game = SnakeApi()
    done = []
    scores = []

    # Increase game speed
    game.game_speed = game.game_speed * 10

    # If weights from previous trains is present, load it
    if os.path.isfile('model_weights.pth'):
        agent.model.load_state_dict(torch.load('model_weights.pth', weights_only=True))

    while agent.n_games < nb_game:
        state_old = game.get_state()

        # 2. L'agent décide de l'action à réaliser
        final_move = agent.get_action(state_old)
        #print(f"Action choisie : {final_move}")

        # 3. On réalise l'action et on récupère la récompense, l'état suivant et l'indicateur de fin de partie
        try:
            reward, done_iteration, score = game.play_step(final_move)
        except EndOfSnakeGame:
            end_of_game(agent, scores)
        
        if play_only:    
            if done_iteration:
                end_of_iteration(agent, game, score, scores)
            continue
        #print(f"Game {agent.n_games} Score {game.score}")
        #print(f"Q-values: {agent.model(torch.tensor(state_old, dtype=torch.float))}")
        #print(f"Action choisie : {final_move}")
        #print(f"Direction actuelle : {game.direction}")
        # print(f"Snake Position: {game.snake}")  # ← Vérifier si elle meurt instantanément
        # print(f"Collision: {game._is_collision()}")  # ← Vérifier pourquoi elle s’arrête
        # print("======")
        # print(f"Utilisation RAM : {psutil.virtual_memory().percent}%")
        # print("======")
    
        state_new = game.get_state()
        done.append(done_iteration)

        # 4. Entraînement à court terme (sur cette transition)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # 5. Stockage de la transition dans la mémoire d'expérience
        agent.remember(state_old, final_move, reward, state_new, done)

        #6 MAJ des logs
        # game_id = "partie_" + str(agent.n_games)
        # log_iteration(game_id, state_old, final_move, reward, done[agent.n_games])

        if done_iteration:
            end_of_iteration(agent, game, score, scores)
            agent.train_long_memory()  # Entraînement sur un mini-batch de la mémoire
            
    end_of_game(agent, scores)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    nb_game = 1
    inference = True
    training = False
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
            training = True
        elif opt in ("-i", "--inference"):
            inference = True
        elif opt in ("-n", "--nb"):
            nb_game = int(arg)

    if training:
        inference = False

    train(nb_game, inference)