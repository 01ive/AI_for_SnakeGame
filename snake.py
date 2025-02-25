import pygame
from enum import Enum
import random
from collections import namedtuple

# Enumération pour la direction du serpent
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Définition d'un point sur la grille
Point = namedtuple('Point', 'x y')

# Initialisation de Pygame et de la police d'écriture
pygame.init()
font = pygame.font.SysFont('arial', 25)

class SnakeGame:
    def __init__(self, width=640, height=480, block_size=20):
        self.width = width
        self.height = height
        self.block_size = block_size
        self._game_speed = 10^5
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
        self.clock.tick(self._game_speed)  # Régule la vitesse du jeu (10 FPS)

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
        return reward, game_over, self.score

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


if __name__ == '__main__':
    game = SnakeGame()
    while True:
        reward, game_over, score = game.play_step()

        if game_over:
            print("Game Over! Score:", score)
            game.reset()
