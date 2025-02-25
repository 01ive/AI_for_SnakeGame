import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import numpy as np
import json
import time
from datetime import datetime as dt
import gc
import os

# Game imports
from snake_api import SnakeApi
from snake import Point
from snake import Direction


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
EPSILON_INI = 60
DATE = str(dt.now().year) + str(dt.now().month) + str(dt.now().day) +  str(dt.now().hour) + str(dt.now().minute) + str(dt.now().second)


def get_state(game):
    head = game.snake[0]
    point_l = Point(head.x - game.block_size, head.y)
    point_r = Point(head.x + game.block_size, head.y)
    point_u = Point(head.x, head.y - game.block_size)
    point_d = Point(head.x, head.y + game.block_size)

    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = [
        # Danger tout droit
        (dir_r and game._is_collision(point_r)) or
        (dir_l and game._is_collision(point_l)) or
        (dir_u and game._is_collision(point_u)) or
        (dir_d and game._is_collision(point_d)),

        # Danger à droite
        (dir_u and game._is_collision(point_r)) or
        (dir_d and game._is_collision(point_l)) or
        (dir_l and game._is_collision(point_u)) or
        (dir_r and game._is_collision(point_d)),

        # Danger à gauche
        (dir_d and game._is_collision(point_r)) or
        (dir_u and game._is_collision(point_l)) or
        (dir_r and game._is_collision(point_u)) or
        (dir_l and game._is_collision(point_d)),

        # Direction actuelle
        dir_r,
        dir_l,
        dir_u,
        dir_d,

        # Position de la nourriture par rapport à la tête
        game.food.x < game.head.x,  # nourriture à gauche
        game.food.x > game.head.x,  # nourriture à droite
        game.food.y < game.head.y,  # nourriture en haut
        game.food.y > game.head.y   # nourriture en bas
    ]

    return np.array(state, dtype=int)

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

class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LinearQNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # Prédiction des Q-valeurs pour l'état actuel
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Pour l'exploration
        self.gamma = 0.9  # Facteur de discount
        self.memory = deque(maxlen=MAX_MEMORY)  # Mémoire d'expérience
        self.model = LinearQNet(input_size=11, hidden_size=128, output_size=3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        return get_state(game)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        t1 = time.time()
        # Epsilon décroissant pour favoriser l'exploitation au fil du temps
        self.epsilon = EPSILON_INI - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        t2 = time.time()
        if t2 - t1 > 0.2 :
            print("Temps de get_action : ", t2-t1)
        return final_move


def train():
    agent = Agent()
    game = SnakeApi()
    done = []

    # Increase game speed
    game.game_speed = game.game_speed * 10

    # If weights from previous trains is present, load it
    if os.path.isfile('model_weights.pth'):
        agent.model.load_state_dict(torch.load('model_weights.pth', weights_only=True))

    while True:
        state_old = agent.get_state(game)

        # 2. L'agent décide de l'action à réaliser
        final_move = agent.get_action(state_old)
        #print(f"Action choisie : {final_move}")

        # 3. On réalise l'action et on récupère la récompense, l'état suivant et l'indicateur de fin de partie
        reward, done_iteration, score = game.play_step(final_move)
        #print(f"Game {agent.n_games} Score {game.score}")
        #print(f"Q-values: {agent.model(torch.tensor(state_old, dtype=torch.float))}")
        #print(f"Action choisie : {final_move}")
        #print(f"Direction actuelle : {game.direction}")
        # print(f"Snake Position: {game.snake}")  # ← Vérifier si elle meurt instantanément
        # print(f"Collision: {game._is_collision()}")  # ← Vérifier pourquoi elle s’arrête
        # print("======")
        # print(f"Utilisation RAM : {psutil.virtual_memory().percent}%")
        # print("======")

        state_new = agent.get_state(game)
        done.append(done_iteration)

        # 4. Entraînement à court terme (sur cette transition)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # 5. Stockage de la transition dans la mémoire d'expérience
        agent.remember(state_old, final_move, reward, state_new, done)

        #6 MAJ des logs
        game_id = "partie_" + str(agent.n_games)
        log_iteration(game_id, state_old, final_move, reward, done[agent.n_games])

        if done_iteration:
            # Réinitialisation du jeu et mise à jour des statistiques
            game.reset()
            # print("========\nGame ended\n=========\n")
            agent.train_long_memory()  # Entraînement sur un mini-batch de la mémoire
            agent.n_games += 1
            print("Objets non collectés :", gc.collect())
            print('Game', agent.n_games, 'Score', score)
            # save weight every 10 games
            if agent.n_games % 10 == 0:
                print('Save weights to model_weights.pth')
                torch.save(agent.model.state_dict(), 'model_weights.pth')


if __name__ == '__main__':
    train()