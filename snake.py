from __future__ import division, print_function
from collections import deque
import pygame
import random



class SnakeGame(object):

    def __init__(self):

        pygame.init()

        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_BLACK = (0, 0, 0)
        self.COLOR_RED = (255, 0, 0)
        self.GAME_SIZE = 80
        self.WIDTH = 10
        self.MAX_TRIES_PER_GAME = 200

    def reset(self):
        self.game_over = False
        self.X = 20
        self.Y = 40
        self.FOOD_X = random.randint(0, (self.GAME_SIZE - self.WIDTH) / self.WIDTH) * self.WIDTH
        self.FOOD_Y = random.randint(0, (self.GAME_SIZE - self.WIDTH) / self.WIDTH) * self.WIDTH
        self.SCORE = 0
        self.REWARD = 0
        self.SNAKE_LIST = []
        self.num_tries = 0
        self.frames = deque(maxlen=2)
        self.action = 0

        self.screen = pygame.display.set_mode((self.GAME_SIZE, self.GAME_SIZE))
        self.clock = pygame.time.Clock()

    def step(self, action):
        self.REWARD = 0
        self.action = action
        self.previous_action = self.action
        self.screen.fill(self.COLOR_BLACK)

        if self.SCORE != 0:
            if self.previous_action == 0 and self.action == 1:
                self.action = self.previous_action
            if self.previous_action == 1 and self.action == 0:
                self.action = self.previous_action
            if self.previous_action == 3 and self.action == 2:
                self.action = self.previous_action
            if self.previous_action == 2 and self.action == 3:
                self.action = self.previous_action

        if self.action == 0:  # up
            self.Y = self.Y - self.WIDTH
        elif self.action == 1:  # down
            self.Y = self.Y + self.WIDTH
        elif self.action == 2:  # right
            self.X = self.X + self.WIDTH
        elif self.action == 3:  # left
            self.X = self.X - self.WIDTH
        else:
            print('Error: Action out of bound')

        self.previous_action = self.action

        if self.X < 0 or self.X > self.GAME_SIZE - self.WIDTH or \
                self.Y < 0 or self.Y > self.GAME_SIZE - self.WIDTH or \
                [self.X, self.Y] in self.SNAKE_LIST:
            self.game_over = True

        if self.X == self.FOOD_X and self.Y == self.FOOD_Y:
            self.SCORE += 1
            self.REWARD = 1
            self.num_tries = 0
            while [self.FOOD_X, self.FOOD_Y] in self.SNAKE_LIST or [self.FOOD_X, self.FOOD_Y] == [self.X, self.Y]:
                self.FOOD_X = random.randint(0, (self.GAME_SIZE - self.WIDTH) / self.WIDTH) * self.WIDTH
                self.FOOD_Y = random.randint(0, (self.GAME_SIZE - self.WIDTH) / self.WIDTH) * self.WIDTH

        self.SNAKE_LIST.append([self.X, self.Y])

        if len(self.SNAKE_LIST) > self.SCORE + 1:
            self.SNAKE_LIST.pop(0)

        self.num_tries += 1
        if self.num_tries >= self.MAX_TRIES_PER_GAME:
            self.game_over = True

        for i in range(len(self.SNAKE_LIST)):
            pygame.draw.rect(self.screen, self.COLOR_WHITE,
                             pygame.Rect(self.SNAKE_LIST[i][0], self.SNAKE_LIST[i][1], self.WIDTH, self.WIDTH))
        pygame.draw.rect(self.screen, self.COLOR_RED, pygame.Rect(self.FOOD_X, self.FOOD_Y, self.WIDTH, self.WIDTH))

        pygame.event.pump()
        pygame.display.update()

        resize_scale = int(self.GAME_SIZE / self.WIDTH)
        resized_screen = pygame.transform.scale(self.screen, (resize_scale, resize_scale))
        scaled_value = pygame.surfarray.array2d(resized_screen)
        scaled_value = scaled_value.astype("float")
        scaled_value[scaled_value == 16711680.0] = 1.0
        scaled_value[scaled_value == 16777215.0] = 0.5

        self.frames.append(scaled_value)

        return self.frames, self.REWARD, self.game_over, self.SCORE