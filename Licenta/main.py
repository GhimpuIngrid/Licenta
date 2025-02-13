# main.py
import time

import numpy as np
import matplotlib.pyplot as plt

import frogger
from Environment import FroggerEnv
from Agent import Agent

import torch

"""
!!!!!!!!!!!!!!!!! IN ACTORS.PY COMENTEAZA LINIA 34 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""


# Funcție principală de antrenament
def train(num_actions, num_episodes, scores, env, agent, scor_final, avg_scores):
    """
    with open("output.txt", "w") as file:
        file.write("Am inceput antrenamentul")"""

    for episode in range(num_episodes):
        """
        with open("output.txt", "a") as file:
            file.write(f"La episodul: {episode}\n")"""

        # print("Ma antrenez")
        '''if episode == 0:
            env.gameApp.state = "PLAYING"
            env.gameApp.draw()'''
        state = env.reset()
        env.gameApp.state = "PLAYING"
        env.gameApp.draw()
        # print("Stare la reset: ")
        # print(state)
        total_reward = 0

        #time.sleep(2)

        counter = 0
        while True:
            counter += 1
            # time.sleep(1)
            #print("Ma joc")
            action = agent.choose_action(agent.prepare_input(state))
            # print("Actiunea: ", action)
            #env.gameApp.draw()
            next_state, reward, done, _ = env.step(action)
            # print(agent.nn.forward(agent.prepare_input(next_state)).detach())
            #env.gameApp.draw()

            '''if np.array_equal(state, next_state) is True:
                print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
            else: print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")'''
            # print("Starea urmatoare: ")
            # print(next_state)

            """
            with open("output.txt", "a") as file:
                file.write(f"Am primit: {reward} puncte\n")
            print(done)"""

            total_reward += reward
            '''frame1 = agent.prepare_input(state)
            frame2 = agent.prepare_input(next_state)

            images = torch.cat([frame1, frame2], dim=0)'''

            agent.store_transition(agent.prepare_input(state), action, reward, agent.prepare_input(next_state), done)

            '''fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            for i, ax in enumerate(axes):
                ax.imshow(images[i, 0].numpy(), cmap='gray')  # Extragem imaginea și o convertim în numpy
                ax.axis('off')

            plt.show()'''

            agent.train()
            state = next_state

            """
            with open("output.txt", "a") as file:
                file.write(f"Am in total: {total_reward} puncte\n")"""
            # print(total_reward)
            # print("Am trecut", done)

            # print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

            '''if counter == 5:
                break'''

            if done:
                #print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
                break

        scores.append(env.highest_lane)

        average_reward = np.mean(scores[-100:])
        avg_scores.append(average_reward)
        # print(1)
        agent.update_target_nn()
        # print(2)
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Avarage reward: {average_reward} Epsilon: {agent.eps}")

    print(scor_final.high_score)
    agent.save_model()
    #env.close()


def test(env, agent):

    test_rewards = []

    for i in range(5):
        agent.load_model()
        state = env.reset()
        total_reward = 0

        env.gameApp.draw()
        env.gameApp.state = "PLAYING"

        while True:

            action = agent.choose_action_test(agent.prepare_input(state))
            print("Am facut actiunea: ", action)
            next_state, reward, done, _ = env.step(action)
            print("Am murit: ", done)
            print("am primit: ", reward)
            time.sleep(0.5)
            env.gameApp.draw()

            ###################################
            state = next_state
            total_reward += reward

            if done:
                print(f"Total Reward: {total_reward}")
                break

        test_rewards.append(total_reward)
        print(f"Episode: {i + 1}, Total Reward: {total_reward}")

    av_reward = sum(test_rewards) / len(test_rewards)
    print("Avarage reward: ", av_reward)

    # env.close()


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


if __name__ == "__main__":
    env = FroggerEnv()

    num_actions = 5
    num_episodes = 8000
    scores = []
    avg_scores = []

    agent = Agent(num_actions, input_dims=[1, 84, 84], batch_size=128)
    scor_final = frogger.Score()

    '''with open("retea.txt", "w") as file:
        file.write("Antrenare\n")'''

    train(num_actions, num_episodes, scores, env, agent, scor_final, avg_scores)

    test(env, agent)

    plt.figure(figsize=(20, 10))
    plt.plot(scores, label="Reward per Episode")
    plt.plot(avg_scores, label="Average of last 100 episodes", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Agent Performance Over Time")
    plt.legend()
    plt.show()

