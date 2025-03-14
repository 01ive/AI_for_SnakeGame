# Standard imports
import os
import logging
# Module imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import deque
# Game imports
from snake_api import EndOfSnakeGame


class LinearQNet(nn.Module):    
    def __init__(self, input_size, hidden_size, output_size):
        super(LinearQNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
class DeepQNet(nn.Module):    
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepQNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
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
    LR = 0.001
    EPSILON_INI = 60
    SPEEDUP_GAME = 100
    MAX_MEMORY = 100_000
    BATCH_SIZE = 1000

    def __init__(self, game, inference=True, nb_game=1, hidden_size=128, class_model=LinearQNet, weight_file_name='model_weights.pth'):
        self._game = game
        self._inference = inference
        self._weight_file_name = weight_file_name
        self._first_training = True
        self._nb_games = nb_game
        self._n_games = 0
        self._scores = []
        self._score = 0
        self._epsilon = 0  # Pour l'exploration
        self._gamma = 0.9  # Facteur de discount
        self._model = class_model(input_size=11, hidden_size=hidden_size, output_size=3)
        self._trainer = QTrainer(self._model, lr=self.LR, gamma=self._gamma)
        self._memory = deque(maxlen=self.MAX_MEMORY)  # Mémoire d'expérience

    def _get_action(self, state):
        # Epsilon décroissant pour favoriser l'exploitation au fil du temps
        self._epsilon = self.EPSILON_INI - self._n_games
        final_move = [0, 0, 0]
        if (random.randint(0, 200) < self._epsilon) and self._first_training:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self._model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move
    
    def _end_of_iteration(self):
        # Réinitialisation du jeu et mise à jour des statistiques
        self._game.reset()
        self._n_games += 1
        self._scores.append(self._score)

        # Entraînement à long terme (sur la mémoire d'expérience)
        if not self._inference:
            if len(self._memory) > self.BATCH_SIZE:
                mini_sample = random.sample(self._memory, self.BATCH_SIZE)
            else:
                mini_sample = self._memory
            states, actions, rewards, next_states, dones = zip(*mini_sample)
            self._trainer.train_step(states, actions, rewards, next_states, dones)

        logging.info('Game {} score {}'.format(self._n_games, self._score))

    def _end_of_game(self):
        # save weight
        if not self._inference:
            logging.debug('Save weights to ' + self._weight_file_name)
            torch.save(self._model.state_dict(), self._weight_file_name)
        logging.info('Mean score: {}'.format(np.array(self._scores).mean()))
        logging.info('Max score: {}'.format(np.array(self._scores).max()))
        logging.info('Standard deviation: {}'.format(np.array(self._scores).std()))

    @property
    def scores(self):
        return self._scores

    def train(self, speed=SPEEDUP_GAME):
        done = []
        
        # Increase game speed
        self._game.game_speed = self._game.game_speed * speed

        # If weights from previous trains is present, load it
        if os.path.isfile(self._weight_file_name):
            self._model.load_state_dict(torch.load(self._weight_file_name, weights_only=True))
            self._first_training = False
        elif self._inference:
            raise FileNotFoundError('No weight file found')

        while self._n_games < self._nb_games:
            state_old = self._game.get_state()

            # 2. L'agent décide de l'action à réaliser
            final_move = self._get_action(state_old)

            # 3. On réalise l'action et on récupère la récompense, l'état suivant et l'indicateur de fin de partie
            try:
                reward, done_iteration, self._score = self._game.play_step(final_move)
            except EndOfSnakeGame:
                break
            
            if self._inference:    
                if done_iteration:
                    self._end_of_iteration()
                continue
                
            state_new = self._game.get_state()
            done.append(done_iteration)

            # 4. Entraînement à court terme (sur cette transition)
            self._trainer.train_step(state_old, final_move, reward, state_new, done)

            # 5. Stockage de la transition dans la mémoire d'expérience
            self._memory.append((state_old, final_move, reward, state_new, done))

            if done_iteration:
                self._end_of_iteration()
                
        self._end_of_game()
