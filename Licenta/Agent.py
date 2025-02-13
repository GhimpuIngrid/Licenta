import torch
import torch.nn as nn
import torch.optim as optim
import NN  # Schimbat: Importul modulului NN care conține rețeaua neuronală actualizată
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# from torch.utils.tensorboard import SummaryWriter


class Agent:
    def __init__(self, number_of_actions, input_dims, batch_size, learning_rate=0.0002, gamma=0.99, initial_epsilon=1.0,
                 epsilon_decay=0.9999, min_epsilon=0.01, max_mem_size=20000):

        self.number_of_actions = number_of_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.eps = initial_epsilon
        self.eps_decay = epsilon_decay
        self.min_eps = min_epsilon

        self.input_dims = input_dims
        self.batch_size = batch_size
        self.mem_size = max_mem_size
        self.mem_cntr = 0

        self.nn = NN.NN(self.number_of_actions, self.lr, self.input_dims)
        self.target_nn = NN.NN(self.number_of_actions, self.lr, self.input_dims)
        self.update_target_nn()

        self.optimizer = optim.Adam(self.nn.parameters(), lr=learning_rate)
        # self.optimizer = optim.SGD(self.nn.parameters(), lr=learning_rate, momentum=0.9)

        self.state_memory = np.zeros((self.mem_size, 84*84), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, 84*84), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

        # self.writer = SummaryWriter("logs/weights")
        self.counter = 0
        self.prev_weights = {name: param.clone().detach() for name, param in self.nn.named_parameters()}

    def store_transition(self, state, action, reward, new_state, done):
        if self.mem_cntr == self.mem_size:
            print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def update_target_nn(self, tau=0.1):
        # self.target_nn.load_state_dict(self.nn.state_dict())
        for target_param, param in zip(self.target_nn.parameters(), self.nn.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def choose_action(self, state):
        if np.random.rand() < self.eps:
            action = np.random.randint(self.number_of_actions)
            # print("RANDOM ", action)
            return action
        else:
            obs = state.to(self.nn.device)
            # print("Observation: ", obs)
            '''if torch.any(obs != 0):
                print("Am valori diferite de 0")
            else:
                print("DOAR ZERO")'''
            with torch.no_grad():
                q_values = self.nn.forward(obs)
            # print(q_values)
            #print("Din choose_action")
            return torch.argmax(q_values).item()

    def choose_action_test(self, state):
        q_values = self.nn.forward(state)
        return torch.argmax(q_values).item()

    def prepare_input(self, frame):
        # print("state inainte de modificare: ")
        # print(frame.tolist())

        frame = Image.fromarray(frame).resize((84, 84))  # Schimbat: Redimensionarea frame-ului la 84x84
        frame = frame.convert('L')  # Schimbat: Conversia frame-ului în scală de gri

        transform = transforms.Compose([transforms.ToTensor()])
        frame = transform(frame)
        # print("state dupa modificare: ")
        # print(frame.tolist())

        return frame.view(-1).unsqueeze(0)  # Schimbat: Adăugarea unei dimensiuni suplimentare pentru batch size????????????????

    def train(self):
        # Vezi batch!!!!!!!!!!!!!!!!!!!!!!!!!
        if self.mem_cntr < self.batch_size:
            return

        self.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.nn.device)

                # Afișăm cele 5 imagini
        '''fig, axes = plt.subplots(1, 5, figsize=(15, 5))
        for i, ax in enumerate(axes):
            ax.imshow(state_batch[i, 0].numpy(), cmap='gray')  # Extragem imaginea și o convertim în numpy
            ax.axis('off')

        plt.show()'''

        # print("In state_batch am tensori diferiti: ", contains_different_tensors(state_batch))
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.nn.device)

        '''fig, axes = plt.subplots(1, 5, figsize=(15, 5))
        for i, ax in enumerate(axes):
            ax.imshow(new_state_batch[i, 0].numpy(), cmap='gray')  # Extragem imaginea și o convertim în numpy
            ax.axis('off')

        plt.show()'''

        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.nn.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.nn.device)

        action_batch = self.action_memory[batch]

        #print("Din train")

        '''with open("weight_changes.txt", "a") as f:
            f.write(f"Epoch {self.counter}:\n")
            for name, param in self.nn.named_parameters():
                diff = torch.abs(param - self.prev_weights[name]).mean().item()
                f.write(f"{name}: diff={diff}\n")
            f.write("\n")'''

        # Actualizează ponderile pentru epoca următoare
        self.prev_weights = {name: param.clone().detach() for name, param in self.nn.named_parameters()}

        q_values = self.nn.forward(state_batch)[batch_index, action_batch]
        # print("Q-Values: \n", q_values)
        next_actions = torch.argmax(self.nn.forward(new_state_batch), dim=1).unsqueeze(1)
        # print("next actions: \n", next_actions)
        next_q_values = self.target_nn.forward(new_state_batch).gather(1, next_actions).squeeze(1).detach()
        # next_q_values[terminal_batch] = 0.0
        # print("next_q_values: \n", next_q_values)

        # ???? 1 - done nu stiu daca e bine
        target = reward_batch + self.gamma * next_q_values
        # print("target: \n", target)
        loss = self.nn.loss(target, q_values).to(self.nn.device)
        # print("loss: \n", loss)

        loss.backward()
        self.optimizer.step()

        self.eps = max(self.min_eps, self.eps * self.eps_decay)
        self.counter += 1

    def save_model(self):
        torch.save(self.nn.state_dict(), "agent.txt")

    def load_model(self):
        self.nn.load_state_dict(torch.load("agent.txt"))
        self.nn.eval()


def contains_different_tensors(t):
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            if not torch.equal(t[i], t[j]):
                return True  # Sub-tensori diferiți găsiți
    return False
