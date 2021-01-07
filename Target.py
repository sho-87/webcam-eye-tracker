import pygame
from utils import get_config

# Read config.ini file
SETTINGS, COLOURS, EYETRACKER, TF = get_config("config.ini")


class Target:
    def __init__(self, pos, speed, radius=10, color=(255, 255, 255)):
        super().__init__()
        self.x = pos[0]
        self.y = pos[1]
        self.speed = speed
        self.radius = radius
        self.color = color
        self.moving = False

    def render(self, screen):
        pygame.draw.circle(
            screen, COLOURS["white"], (self.x, self.y), self.radius + 1, 0
        )
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius, 0)

    def move(self, target_loc, ticks):
        dist_per_tick = self.speed * ticks / 1000

        if (
            abs(self.x - target_loc[0]) <= dist_per_tick
            and abs(self.y - target_loc[1]) <= dist_per_tick
        ):
            self.moving = False
            self.color = COLOURS["red"]
        else:
            self.moving = True
            self.color = COLOURS["green"]
            current_vector = pygame.Vector2((self.x, self.y))
            new_vector = pygame.Vector2(target_loc)
            towards = (new_vector - current_vector).normalize()

            self.x += towards[0] * dist_per_tick
            self.y += towards[1] * dist_per_tick
