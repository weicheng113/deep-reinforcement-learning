from model import Actor
from model import Critic
import torch
from soccer_env import SoccerEnvWrapper
from unityagents import UnityEnvironment
from train import train, plot_scores
from multi_agent import MultiAgent, PPOAgent
import os
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 2


def run_training():
    soccer_env = UnityEnvironment(file_name="Soccer_Linux/Soccer.x86_64")
    env = SoccerEnvWrapper(env=soccer_env, train_mode=True)

    def create_agents():
        agents = []
        for _ in range(env.num_goalies):
            agents.append(create_agent(env, action_dim=env.goalie_action_size))

        for _ in range(env.num_strikers):
            agents.append(create_agent(env, action_dim=env.striker_action_size))

        return agents

    multi_agent = MultiAgent(
        agents=create_agents(),
        device=device,
        discount=0.99,
        seed=seed)
    scores = train(env=env, agent=multi_agent, episodes=5000, max_t=1000)
    plot_scores(scores)


def create_agent(env, action_dim):
    learning_rate = 1e-4
    batch_size = 128

    def create_actor():
        return Actor(
            state_dim=env.state_size,
            action_dim=action_dim,
            fc1_units=400,
            fc2_units=300,
            seed=seed)

    def create_critic():
        return Critic(
            state_dim=env.state_size * env.num_agents,
            fc1_units=400,
            fc2_units=300,
            seed=seed)

    agent = PPOAgent(
        create_actor=lambda: create_actor(),
        create_critic=create_critic,
        state_dim=env.state_size,
        optimization_epochs=4,
        batch_size=batch_size,
        epsilon=0.1,
        entropy_weight=0.01,
        lr=learning_rate,
        device=device,
        seed=seed)
    return agent


def test():
    soccer_env = UnityEnvironment(file_name="Soccer_Linux/Soccer.x86_64")
    env = SoccerEnvWrapper(env=soccer_env, train_mode=False)

    def create_agents():
        agents = []
        agents.append(create_random_agent(env, action_dim=env.goalie_action_size))
        goalie = create_agent(env, action_dim=env.goalie_action_size)
        goalie.actor.load_state_dict(torch.load(f"checkpoints/checkpoint_actor_1_episode_3000.pth"))
        goalie.critic.load_state_dict(torch.load(f"checkpoints/checkpoint_critic_1_episode_3000.pth"))
        agents.append(goalie)

        agents.append(create_random_agent(env, action_dim=env.striker_action_size))
        striker = create_agent(env, action_dim=env.striker_action_size)
        striker.actor.load_state_dict(torch.load(f"checkpoints/checkpoint_actor_3_episode_3000.pth"))
        striker.critic.load_state_dict(torch.load(f"checkpoints/checkpoint_critic_3_episode_3000.pth"))
        agents.append(striker)

        return agents

    agent = MultiAgent(
        agents=create_agents(),
        device=device,
        discount=0.99,
        seed=seed)

    for i_episode in range(10):
        states = env.reset()
        agent.reset()
        score = np.zeros(env.num_agents)
        steps = 0

        for t in range(1000):
            actions, action_probs = agent.act(states)
            # print(f"actions: {actions}, action_probs: {action_probs}")
            next_states, rewards, dones = env.step(actions)
            agent.step(states, actions, action_probs, rewards, next_states, dones)
            states = next_states

            score += rewards
            steps += 1
            if np.any(dones):
                break
        print(f"Episode {i_episode}\t",
              f"Score: {score}\t",
              f"Steps: {steps}\t")


def create_random_agent(env, action_dim):
    class RandomAgent:
        def __init__(self):
            self.action_dim = action_dim

        def act(self, state):
            prob = 0.5
            return np.random.randint(self.action_dim), prob

        def reset(self):
            pass

    return RandomAgent(action_dim=action_dim)


if __name__ == '__main__':
    directory = 'checkpoints'
    if not os.path.exists(directory):
        os.makedirs(directory)
    run_training()
    # test()


