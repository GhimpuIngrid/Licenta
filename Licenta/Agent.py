import torch
import torch.nn as nn
import torch.optim as optim
import NN  # Schimbat: Importul modulului NN care conține rețeaua neuronală actualizată
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class Agent:
    def __init__(self, number_of_actions, learning_rate=0.001, gamma=0.95, initial_epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.number_of_actions = number_of_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.eps = initial_epsilon
        self.eps_decay = epsilon_decay
        self.min_eps = min_epsilon
        self.nn = NN.NN(number_of_actions)  # Schimbat: Inițializarea rețelei neuronale cu numărul de acțiuni
        self.optimizer = optim.Adam(self.nn.parameters(), lr=learning_rate)

    def choose_action(self, state):
        if np.random.rand() < self.eps:
            return np.random.randint(self.number_of_actions)
        else:
            q_values = self.nn.forward(state)
            return torch.argmax(q_values).item()

    def prepare_input(self, frame):
        frame = Image.fromarray(frame).resize((84, 84))  # Schimbat: Redimensionarea frame-ului la 84x84
        frame = frame.convert('L')  # Schimbat: Conversia frame-ului în scală de gri

        transform = transforms.Compose([transforms.ToTensor()])
        frame = transform(frame)
        return frame.unsqueeze(0)  # Schimbat: Adăugarea unei dimensiuni suplimentare pentru batch size

    def train(self, state, action, reward, next_state, done):
        q_values = self.nn(state)
        #print(q_values)
        next_q_values = self.nn(next_state)

        target = reward + (1 - done) * self.gamma * torch.max(next_q_values)
        #print(target)
        loss = nn.functional.mse_loss(q_values[0][action], target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.eps = max(self.min_eps, self.eps * self.eps_decay)

    def save_model(self):
        torch.save(self.nn.state_dict(), "agent.txt")

    def load_model(self):
        self.nn.load_state_dict(torch.load("agent.txt"))
        self.nn.eval()
