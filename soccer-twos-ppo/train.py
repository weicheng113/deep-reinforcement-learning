import progressbar as pb
import numpy as np
# from collections import deque
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import psutil
import time


def train(env, agent, episodes=1000, max_t=1000, print_every=50):
    widget = ['training loop: ', pb.Percentage(), ' ',  pb.Bar(), ' ', pb.ETA()]
    timer = pb.ProgressBar(widgets=widget, maxval=episodes).start()

    scores = []
    # scores_deque = deque(maxlen=100)
    # steps_deque = deque(maxlen=100)
    for i_episode in range(1, episodes+1):
        states = env.reset()
        agent.reset()
        score = np.zeros(env.num_agents)
        steps = 0
        start = time.time()

        for t in range(max_t):
            actions, action_probs = agent.act(states)
            next_states, rewards, dones = env.step(actions)
            agent.step(states, actions, action_probs, rewards, next_states, dones)
            states = next_states

            score += rewards
            steps += 1
            if np.any(dones):
                break

            if (t+1) % 5 == 0:
                agent.episode_done(i_episode)
        agent.episode_done(i_episode)

        # scores_deque.append(np.max(score))
        scores.append(np.max(score))
        # steps_deque.append(steps)
        time_spent = time.time() - start
        print(f"Episode {i_episode}/{episodes}\t",
              f"Score: {score}\t",
              f"Steps: {steps}\t",
              f"Memory usage: {memory_usage_in_megabytes():.2f}MB\t",
              f"Time spent: {time_spent} seconds")

        if i_episode % print_every == 0:
            print()
            timer.update(i_episode)

        if i_episode % 500 == 0:
            print(f"saving model at {i_episode}")
            for i, agent_i in enumerate(agent.agents):
                torch.save(agent_i.actor.state_dict(), f"checkpoints/checkpoint_actor_{i}_episode_{i_episode}.pth")
                torch.save(agent_i.critic.state_dict(), f"checkpoints/checkpoint_critic_{i}_episode_{i_episode}.pth")

        #         if (scores_deque[0]>0.5) and (np.mean(scores_deque) > 0.5):
        # if np.mean(scores_deque) > 0.5:
        #     print(f"Environment solved in {i_episode-100} episodes!\t Average Score: {np.mean(scores_deque):.2f}")
        #     for i, agent in enumerate(agent.agents):
        #         torch.save(agent.actor.state_dict(), f"checkpoint_actor_{str(i)}.pth")
        #         torch.save(agent.critic.state_dict(), f"checkpoint_critic_{str(i)}.pth")
        #     break

    timer.finish()
    return scores


def memory_usage_in_megabytes():
    process = psutil.Process(os.getpid())
    current_memory = process.memory_info().rss
    return current_memory/(1024.0*1024.0)


def plot_scores(scores):
    sns.set(style="whitegrid")
    episodes = np.arange(start=1, stop=len(scores)+1)

    data = pd.DataFrame(data=scores, index=episodes, columns=["Score"])

    fig = sns.lineplot(data=data)
    fig.set_xlabel("Episode #")
    plt.show()
