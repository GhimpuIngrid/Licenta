# main.py
import time

import frogger
from Environment import FroggerEnv
from Agent import Agent

"""
!!!!!!!!!!!!!!!!! IN ACTORS.PY COMENTEAZA LINIA 34 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

# Configurăm parametrii pentru agent
num_actions = 5
num_episodes = 1000

# Inițializăm environment-ul Frogger și agentul
env = FroggerEnv()
agent = Agent(num_actions)
scor_final = frogger.Score()

# Funcție principală de antrenament
def train():
    """
    with open("output.txt", "w") as file:
        file.write("Am inceput antrenamentul")"""

    for episode in range(num_episodes):
        """
        with open("output.txt", "a") as file:
            file.write(f"La episodul: {episode}\n")"""

        #print("Ma antrenez")
        state = env.reset()
        total_reward = 0

        while True:
            #print("Ma joc")
            action = agent.choose_action(agent.prepare_input(state))
            next_state, reward, done, _ = env.step(action)

            """
            with open("output.txt", "a") as file:
                file.write(f"Am primit: {reward} puncte\n")
            print(done)"""

            total_reward += reward
            agent.train(agent.prepare_input(state), action, total_reward, agent.prepare_input(next_state), done)
            state = next_state

            """
            with open("output.txt", "a") as file:
                file.write(f"Am in total: {total_reward} puncte\n")"""
            # print(total_reward)
            # print("Am trecut", done)
            # env.render()

            if done:
                print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
                break

        # print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.eps}")

    print(scor_final.high_score)
    #env.close()


def test():
    state = env.reset()
    total_reward = 0

    env.gameApp.draw()
    env.gameApp.state = "PLAYING"

    while True:

        action = agent.choose_action_test(agent.prepare_input(state))
        next_state, reward, done, _ = env.step(action)
        time.sleep(0.5)
        env.gameApp.draw()

        ###################################
        state = next_state
        total_reward += reward

        if done:
            print(f"Total Reward: {total_reward}")
            break

    env.close()



if __name__ == "__main__":
    train()
    test()
