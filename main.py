import pygame
from enum import Enum
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import numpy as np
import json
import time
# import psutil
# import datetime
from datetime import datetime as dt
import gc


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
EPSILON_INI = 60
DATE = str(dt.now().year) + str(dt.now().month) + str(dt.now().day) +  str(dt.now().hour) + str(dt.now().minute) + str(dt.now().second)
GameSpeed = 10^5


# Initialisation de Pygame et de la police d'écriture
pygame.init()
font = pygame.font.SysFont('arial', 25)


# Enumération pour la direction du serpent
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


# Définition d'un point sur la grille
Point = namedtuple('Point', 'x y')


class SnakeGame:
    def __init__(self, width=640, height=480, block_size=20):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.reset()  # Initialisation de l'état du jeu

    def reset(self):
        """
        Réinitialise l'état du jeu : position du serpent, score, nourriture...
        """
        self.direction = Direction.RIGHT
        self.head = Point(self.width // 2, self.height // 2)
        self.headprev = None
        self.snake = [
            self.head,
            Point(self.head.x - self.block_size, self.head.y),
            Point(self.head.x - 2 * self.block_size, self.head.y)
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        # Actualiser l'interface graphique
        self._update_ui()
        self.clock.tick(GameSpeed)  # Régule la vitesse du jeu (10 FPS)

    def _place_food(self):
        """
        Place la nourriture aléatoirement sur la grille,
        en s'assurant qu'elle n'apparaisse pas sur le corps du serpent.
        """
        x = random.randint(0, (self.width - self.block_size) // self.block_size) * self.block_size
        y = random.randint(0, (self.height - self.block_size) // self.block_size) * self.block_size
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def is_closer_to_food(self):
        prev_distance = abs(self.headprev.x - self.food.x) + abs(self.headprev.y - self.food.y)
        new_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        return new_distance < prev_distance  # True si le serpent s'est rapproché

    def play_step(self, action):
        """
        Joue un tour du jeu basé sur une action donnée par l'IA.

        Paramètres :
          - action (list) : Un tableau de 3 éléments [0, 1, 0] indiquant la direction choisie.

        Retourne :
          - reward (int) : La récompense obtenue lors de l'action.
          - game_over (bool) : True si la partie est terminée.
          - score (int) : Le score actuel.
        """
        self.frame_iteration += 1

        # Déterminer la nouvelle direction en fonction de l'action choisie par l'IA
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if action == [1, 0, 0]:  # Tourner à gauche
            new_idx = (idx - 1) % 4
        elif action == [0, 1, 0]:  # Continuer tout droit
            new_idx = idx
        else:  # action == [0, 0, 1] → Tourner à droite
            new_idx = (idx + 1) % 4

        self.direction = clock_wise[new_idx]

        # Mettre à jour la position de la tête
        self.headprev = self.head
        self._move()
        self.snake.insert(0, self.head)

        # Vérifier les collisions ou une boucle infinie (pour éviter que le serpent tourne en rond trop longtemps)
        reward = 0
        game_over = False
        if self._is_collision(): #or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Vérifier si la nourriture a été consommée
        if self.is_closer_to_food():
            reward = 5
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()  # Supprimer la queue pour faire avancer le serpent

        # Actualiser l'interface graphique
        self._update_ui()
        self.clock.tick(GameSpeed)  # Régule la vitesse du jeu (10 FPS)
        pygame.event.pump() #Vider la pile ou jsp quoi empêche que la fenêtre plante.
        return reward, game_over, self.score
    """
    def play_step(self):
        self.frame_iteration += 1

        # Gestion des événements clavier
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP and self.direction != Direction.DOWN:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
                    self.direction = Direction.DOWN

        # Mettre à jour la position de la tête en fonction de la direction actuelle
        self._move()
        self.snake.insert(0, self.head)

        # Vérifier la collision ou un temps d'épisode trop long (pour éviter que le joueur ne reste bloqué)
        reward = 0
        game_over = False
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Vérifier si la nourriture a été consommée
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()  # Faire avancer le serpent en supprimant la queue

        # Actualiser l'interface graphique
        self._update_ui()
        self.clock.tick(10)  # Régule la vitesse du jeu (10 FPS)
        return reward, game_over, self.score"""

    def _is_collision(self, pt=None):
        """
        Vérifie si le point donné (par défaut la tête) entre en collision avec
        le bord de l'écran ou avec le corps du serpent.
        """
        if pt is None:
            pt = self.head
        # Collision avec les murs
        if pt.x >= self.width or pt.x < 0 or pt.y >= self.height or pt.y < 0:
            return True
        # Collision avec soi-même
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        """
        Met à jour l'affichage du jeu : serpent, nourriture et score.
        """
        self.display.fill((0, 0, 0))  # Fond noir
        # Dessiner le serpent
        for pt in self.snake:
            pygame.draw.rect(self.display, (0, 255, 0),
                             pygame.Rect(pt.x, pt.y, self.block_size, self.block_size))
            pygame.draw.rect(self.display, (0, 200, 0),
                             pygame.Rect(pt.x + 4, pt.y + 4, self.block_size - 8, self.block_size - 8))
        # Dessiner la nourriture
        pygame.draw.rect(self.display, (255, 0, 0),
                         pygame.Rect(self.food.x, self.food.y, self.block_size, self.block_size))
        # Afficher le score
        text = font.render("Score: " + str(self.score), True, (255, 255, 255))
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self):
        """
        Met à jour la position de la tête du serpent en fonction de la direction actuelle.
        """
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += self.block_size
        elif self.direction == Direction.LEFT:
            x -= self.block_size
        elif self.direction == Direction.DOWN:
            y += self.block_size
        elif self.direction == Direction.UP:
            y -= self.block_size

        self.head = Point(x, y)

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
    game = SnakeGame()
    done = []

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
if __name__ == '__main__':
    train()
    """game = SnakeGame()
    while True:
        reward, game_over, score = game.play_step()

        if game_over:
            print("Game Over! Score:", score)
            game.reset()"""
