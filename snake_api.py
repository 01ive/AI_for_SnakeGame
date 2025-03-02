import pygame
import numpy as np

from snake import Point, Direction, SnakeGame

class EndOfSnakeGame(Exception):
    pass

class SnakeApi(SnakeGame):
    def __init__(self, width=640, height=480, block_size=20):
        super().__init__(width, height, block_size)

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

        # Gestion des événements clavier
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise EndOfSnakeGame

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
        if self._is_closer_to_food():
            reward = 5
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()  # Supprimer la queue pour faire avancer le serpent

        # Actualiser l'interface graphique
        self._update_ui()
        self.clock.tick(self._game_speed)  # Régule la vitesse du jeu (10 FPS)
        pygame.event.pump() #Vider la pile ou jsp quoi empêche que la fenêtre plante.
        return reward, game_over, self.score
        
    def _is_closer_to_food(self):
        prev_distance = abs(self.headprev.x - self.food.x) + abs(self.headprev.y - self.food.y)
        new_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        return new_distance < prev_distance  # True si le serpent s'est rapproché

    def get_state(self):
        head = self.snake[0]
        point_l = Point(head.x - self.block_size, head.y)
        point_r = Point(head.x + self.block_size, head.y)
        point_u = Point(head.x, head.y - self.block_size)
        point_d = Point(head.x, head.y + self.block_size)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # Danger tout droit
            (dir_r and self._is_collision(point_r)) or
            (dir_l and self._is_collision(point_l)) or
            (dir_u and self._is_collision(point_u)) or
            (dir_d and self._is_collision(point_d)),

            # Danger à droite
            (dir_u and self._is_collision(point_r)) or
            (dir_d and self._is_collision(point_l)) or
            (dir_l and self._is_collision(point_u)) or
            (dir_r and self._is_collision(point_d)),

            # Danger à gauche
            (dir_d and self._is_collision(point_r)) or
            (dir_u and self._is_collision(point_l)) or
            (dir_r and self._is_collision(point_u)) or
            (dir_l and self._is_collision(point_d)),

            # Direction actuelle
            dir_r,
            dir_l,
            dir_u,
            dir_d,

            # Position de la nourriture par rapport à la tête
            self.food.x < self.head.x,  # nourriture à gauche
            self.food.x > self.head.x,  # nourriture à droite
            self.food.y < self.head.y,  # nourriture en haut
            self.food.y > self.head.y   # nourriture en bas
        ]

        return np.array(state, dtype=int)
    
    @property
    def game_speed(self):
        return self._game_speed
    
    @game_speed.setter
    def game_speed(self, value):
        self._game_speed = value
