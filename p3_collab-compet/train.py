import progressbar as pb
import numpy as np
from collections import deque
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time


def train(env, agent, episodes=5000, max_t=500, print_every=50):
    widget = ['training loop: ', pb.Percentage(), ' ',  pb.Bar(), ' ', pb.ETA()]
    timer = pb.ProgressBar(widgets=widget, maxval=episodes).start()

    scores = []
    scores_deque = deque(maxlen=100)
    steps_deque = deque(maxlen=100)
    for i_episode in range(1, episodes+1):
        states = env.reset()
        agent.reset()
        score = np.zeros(env.num_agents)
        steps = 0
        start = time.time()

        for t in range(max_t):
            actions = agent.act(states, add_noise=True)
            next_states, rewards, dones = env.step(actions)
            agent.step(i_episode, states, actions, rewards, next_states, dones)
            states = next_states

            score += rewards
            steps += 1
            if np.any(dones):
                break

        scores_deque.append(np.max(score))
        scores.append(np.max(score))
        steps_deque.append(steps)

        time_spent = time.time() - start
        print(f"Episode {i_episode}/{episodes}\t",
              f"Score: {np.max(score):.2f}\t",
              f"Steps: {steps}\t",
              f"Average Score: {np.mean(scores_deque):.2f}\t",
              f"Max Score: {np.max(scores_deque):.2f}\t",
              f"Average steps: {np.mean(steps_deque):.2f}\t",
              f"Time spent: {time_spent:.4f} seconds")
        if i_episode % print_every == 0:
            print()
            timer.update(i_episode)

        #         if (scores_deque[0]>0.5) and (np.mean(scores_deque) > 0.5):
        if np.mean(scores_deque) > 0.5:
            print(f"Environment solved in {i_episode-100} episodes!\t Average Score: {np.mean(scores_deque):.2f}")
            for i, agent in enumerate(agent.agents):
                torch.save(agent.actor.state_dict(), f"checkpoint_actor_{str(i)}.pth")
                torch.save(agent.critic.state_dict(), f"checkpoint_critic_{str(i)}.pth")
            break

    timer.finish()
    return scores


def plot_scores(scores):
    sns.set(style="whitegrid")
    episodes = np.arange(start=1, stop=len(scores)+1)

    data = pd.DataFrame(data=scores, index=episodes, columns=["Score"])

    fig = sns.lineplot(data=data)
    fig.set_xlabel("Episode #")
    plt.show()
