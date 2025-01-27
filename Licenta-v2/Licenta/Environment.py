import pygame
import numpy as np
from pygame.locals import *
from frogger import App
from config import g_vars  # Importing g_vars from config.py

class FroggerEnv:
    def __init__(self):
        self.lives_before = 1
        self.gameApp = App()
        self.action_space = 5
        self.last_action = None
        self.observation_space = (g_vars['width'], g_vars['height'], 3)
        self.highest_lane = None
        self.reset()

    def reset(self):
        self.gameApp.init()
        state = self.get_state()
        return state

    def step(self, action):
        self.last_action = action
        pygame.event.pump()
        if action == 0:
            """with open("output.txt", "a") as file:
                file.write(f"M-am dus sus\n")"""
            self.gameApp.frog.move(0, -1)  # Sus
        elif action == 1:
            """with open("output.txt", "a") as file:
                file.write(f"M-am dus jos\n")"""
            self.gameApp.frog.move(0, 1)  # Jos
        elif action == 2:
            """with open("output.txt", "a") as file:
                file.write(f"M-am dus in stanga\n")"""
            self.gameApp.frog.move(-1, 0)  # Stânga
        elif action == 3:
            """with open("output.txt", "a") as file:
                file.write(f"M-am dus in dreapta\n")"""
            self.gameApp.frog.move(1, 0)  # Dreapta
        elif action == 4:
            """with open("output.txt", "a") as file:
                file.write(f"Am stat pe loc\n")"""
            pass  # Sta pe loc

        self.gameApp.update()
        next_state = self.get_state()
        reward = self.get_reward()
        done = self.is_done()
        # print("Vieti in frogger: ", self.gameApp.score.lives)
        # print("Done in frogger: ", done)
        self.lives_before = self.gameApp.score.lives
        return next_state, reward, done, {}

    def render(self):
        self.gameApp.draw()
        self.gameApp.clock.tick(g_vars['fps'])

    def get_state(self):
        state = pygame.surfarray.array3d(g_vars['window'])
        state = np.transpose(state, (1, 0, 2))
        return state

    def get_reward(self):
        reward = 0
        current_lane = self.gameApp.frog.y  # Rândul curent al broaștei
        # print(f"linia la care sunt: {self.gameApp.current_lane} si inainte eram la linia {self.gameApp.prev_lane}")
        # print(f"Rezultat if: {current_lane < self.gameApp.score.high_lane}")

        # Agentul a pierdut o viață
        if self.gameApp.score.lives < self.lives_before:
            reward -= 100  # Penalizare mare pentru pierderea unei vieți

        elif self.gameApp.current_lane == 12:
            reward += 100

        # Agentul a avansat o linie
        elif self.last_action == 0:
            # print("Am primit punct")
            reward += 20  # Recompensă mare pentru progres

        elif self.gameApp.current_lane < self.gameApp.prev_lane:
            reward -= 5  # Penalizam pentru intoarcere

        # Agentul a ales să stea pe loc
        elif self.last_action == 4:
            reward -= 2  # Penalizare mică pentru inactivitate

        # Agentul se deplasează lateral fără progres
        elif self.last_action in [2, 3]:
            reward -= 1  # Penalizare mică pentru mișcare fără progres

        return reward

    def is_done(self):
        # print("Vieti:", self.gameApp.score.lives)
        """
        with open("output.txt", "a") as file:
            file.write(f"Mai am {self.gameApp.score.lives} vieti\n")"""
        return self.gameApp.score.lives == 0  # Episodul se termină când broasca rămâne fără vieți

    def close(self):
        pygame.quit()
