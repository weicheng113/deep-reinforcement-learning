import progressbar as pb
import numpy as np
from collections import deque
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import psutil
import gc
# from memory_profiler import profile
# import time
# from pympler import tracker
# import gc
# from pympler import muppy
# import objgraph


# @profile
def train(envs, agent, episodes=5000, max_t=1000, print_every=50):
    widget = ['training loop: ', pb.Percentage(), ' ',  pb.Bar(), ' ', pb.ETA()]
    timer = pb.ProgressBar(widgets=widget, maxval=episodes).start()

    scores = []
    scores_deque = deque(maxlen=100)
    steps_deque = deque(maxlen=100)
    # tr = tracker.SummaryTracker()
    process = psutil.Process(os.getpid())
    gc.collect()
    prev_mem = process.memory_info().rss
    for i_episode in range(1, episodes+1):
        score, steps = an_episode(envs=envs, agent=agent, i_episode=i_episode, max_t=max_t)
        # results = envs.reset()
        # states, frame = extract_reset_results(results)
        # score = np.zeros(envs.size)
        # steps = 0
        #
        # for t in range(max_t):
        #     actions, action_probs = agent.act(states)
        #     results = envs.step(actions)
        #     next_states, rewards, dones, info, frames = extract_results(results)
        #     agent.step(
        #         i_episode=i_episode,
        #         states=states,
        #         actions=actions,
        #         action_probs=action_probs,
        #         rewards=rewards,
        #         next_states=next_states,
        #         dones=dones)
        #     states = next_states
        #
        #     score += rewards
        #     steps += 1
        #     if np.any(dones):
        #         break
        scores_deque.append(np.mean(score))
        scores.append(np.mean(score))
        steps_deque.append(steps)

        # print(f"\rEpisode {i_episode}/{episodes}\
        print(f"Episode {i_episode}/{episodes} \
            Score: {score}\
            Average Score: {np.mean(scores_deque):.2f}\
            Max Score: {np.max(scores_deque):.2f}\
            Average steps: {np.mean(steps_deque):.2f}") #, end="")
        if i_episode % print_every == 0:
            print()
            timer.update(i_episode)

        #         if (scores_deque[0]>0.5) and (np.mean(scores_deque) > 0.5):
        if np.mean(scores_deque) > 5:
            print(f"Environment solved in {i_episode-100} episodes!\t Average Score: {np.mean(scores_deque):.2f}")
            torch.save(agent.actor.state_dict(), f"checkpoint_actor.pth")
            torch.save(agent.critic.state_dict(), f"checkpoint_critic.pth")
            break

        gc.collect()
        cur_mem = process.memory_info().rss
        cur_mem_M = cur_mem/(1024.0*1024.0)
        add_mem = (cur_mem - prev_mem) / (1024.0 * 1024.0)
        prev_mem = cur_mem
        print(f"Episode {i_episode}/{episodes}, current mem: {cur_mem_M}M, added mem: {add_mem:.2f}M")
        # gc.collect()
        # tr.print_diff()
        if cur_mem_M > 10000.0:
            print("reach memory limit. saving model")
            torch.save(agent.actor.state_dict(), f"checkpoint_actor1.pth")
            torch.save(agent.critic.state_dict(), f"checkpoint_critic1.pth")
            memReport()
            break

        if i_episode == 100:
            print("debug")

    timer.finish()
    return scores


def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())


# @profile
def an_episode(envs, agent, i_episode, max_t):
    results = envs.reset()
    # time.sleep(5)
    states, frame = extract_reset_results(results)
    score = np.zeros(envs.size)
    steps = 0
    for t in range(max_t):
        actions, action_probs = agent.act(states)
        results = envs.step(actions)
        next_states, rewards, dones, info, frames = extract_results(results)
        agent.step(
            i_episode=i_episode,
            states=states,
            actions=actions,
            action_probs=action_probs,
            rewards=rewards,
            next_states=next_states,
            dones=dones)
        states = next_states

        score += rewards
        steps += 1
        if np.any(dones):
            break
    return score, steps


def extract_reset_results(results):
    states = []
    frames = []
    for result in results:
        state, frame = result
        states.append(state)
        frames.append(frame)
    return states, frames


def extract_results(results):
    states = []
    rewards = []
    dones = []
    infos = []
    frames = []
    for result in results:
        state, reward, done, info, frame = result
        states.append(state)
        rewards.append(reward)
        dones.append(done)
        infos.append(info)
        frames.append(frame)

    return states, rewards, dones, infos, frames


def plot_scores(scores):
    sns.set(style="whitegrid")
    episodes = np.arange(start=1, stop=len(scores)+1)

    data = pd.DataFrame(data=scores, index=episodes, columns=["Score"])

    fig = sns.lineplot(data=data)
    fig.set_xlabel("Episode #")
    plt.show()
