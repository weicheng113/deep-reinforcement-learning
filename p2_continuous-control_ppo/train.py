import progressbar as pb
import torch
import time
from sys_monitor import memory_usage_in_megabytes
from ppo_agent import PPOAgent
from model import Actor, Critic
import numpy as np
from collections import deque


def train(env, agent, episodes, max_t, print_every, logger, checkpoints_dir):
    widget = ['training loop: ', pb.Percentage(), ' ',  pb.Bar(), ' ', pb.ETA()]
    timer = pb.ProgressBar(widgets=widget, maxval=episodes).start()

    scores_deque = deque(maxlen=100)
    for i_episode in range(1, episodes+1):
        states = env.reset()
        agent.reset()
        score = np.zeros(env.num_agents)
        start = time.time()

        for t in range(max_t):
            actions, action_probs = agent.act(states)
            next_states, rewards, dones, info = env.step(actions)
            agent.step(states, actions, action_probs, rewards, next_states, dones)
            states = next_states

            score += rewards
            logger.add_histogram("states", states[0], i_episode)
            logger.add_histogram("rewards", rewards[0], i_episode)
            logger.add_histogram("actions", actions[0], i_episode)
            if np.any(dones):
                break
            # time.sleep(0.5)
        agent.episode_done(i_episode)

        logger.add_histogram("scores", score, i_episode)
        scores_deque.append(np.mean(score))
        time_spent = time.time() - start
        print(f"Episode {i_episode}/{episodes}\t",
              f"Average Score: {np.mean(score):.2f}\t",
              f"Last 100 Average score: {np.mean(scores_deque):.2f}\t",
              f"Memory usage: {memory_usage_in_megabytes():.2f}MB\t",
              f"Time spent: {time_spent} seconds")

        if i_episode % print_every == 0:
            print()
            timer.update(i_episode)

        if np.mean(scores_deque) > 30:
            print(f"\nEnvironment solved in {i_episode} episodes!\t Average Score: {np.mean(scores_deque):.2f}")
            torch.save(agent.actor_local.state_dict(), f"{checkpoints_dir}/checkpoint_actor.pth")
            torch.save(agent.critic_local.state_dict(),f"{checkpoints_dir}/checkpoint_critic.pth")
        logger.add_scalar(f"data/score", np.mean(score), i_episode)

    timer.finish()


def create_agent(env, learning_rate, batch_size, discount,
                 clip_ratio, optimization_epochs, value_loss_weight,
                 entropy_weight, entropy_reduction_rate, max_grad_norm, device,
                 seed, logger):
    def create_actor():
        return Actor(
            state_size=env.state_size,
            action_size=env.action_size,
            seed=seed).to(device)

    def create_critic():
        return Critic(
            state_size=env.state_size,
            seed=seed).to(device)

    agent = PPOAgent(
        create_actor=create_actor,
        create_critic=create_critic,
        state_size=env.state_size,
        num_agents=env.num_agents,
        optimization_epochs=optimization_epochs,
        batch_size=batch_size,
        discount=discount,
        clip_ratio=clip_ratio,
        value_loss_weight=value_loss_weight,
        entropy_weight=entropy_weight,
        entropy_reduction_rate=entropy_reduction_rate,
        lr=learning_rate,
        max_grad_norm=max_grad_norm,
        device=device,
        seed=seed,
        logger=logger)
    return agent
