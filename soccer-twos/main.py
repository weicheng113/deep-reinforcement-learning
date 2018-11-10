from maddpg_agent import MultiAgent
from model import Actor
from model import Critic
from replay_buffer import ReplayBuffer
from noise import RandomUniformNoise
import torch
from soccer_env import SoccerEnvWrapper
from unityagents import UnityEnvironment
from train import train, plot_scores
from maddpg_agent import Agent


def run():
    soccer_env = UnityEnvironment(file_name="Soccer.app")
    env = SoccerEnvWrapper(env=soccer_env, train_mode=True)

    buffer_size = int(1e5)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learning_rate_actor = 1e-4
    learning_rate_critic = 1e-3
    seed = 2
    episodes_before_train = 1
    batch_size = 128
    tau = 1e-3
    full_action_dim = env.goalie_action_size * env.num_goalies + env.striker_action_size * env.num_strikers

    def create_actor(action_dim):
        return Actor(
            state_dim=env.state_size,
            action_dim=action_dim,
            fc1_units=256,
            fc2_units=128,
            seed=seed)

    def create_critic():
        return Critic(
            state_dim=env.state_size * env.num_agents,
            action_dim=full_action_dim,
            fc1_units=256,
            fc2_units=128,
            seed=seed)

    def create_noise(action_dim):
        return RandomUniformNoise(size=action_dim, seed=seed)

    def create_agent(action_dim):
        noise = create_noise(action_dim=action_dim)
        agent = Agent(
            create_actor=lambda: create_actor(action_dim),
            create_critic=create_critic,
            state_dim=env.state_size,
            noise=noise,
            device=device,
            lr_actor=learning_rate_actor,
            lr_critic=learning_rate_critic,
            tau=tau,
            seed=seed)
        return agent

    def create_agents():
        agents = []
        for _ in range(env.num_goalies):
            agents.append(create_agent(action_dim=env.goalie_action_size))

        for _ in range(env.num_strikers):
            agents.append(create_agent(action_dim=env.striker_action_size))

        return agents

    replay_buffer = ReplayBuffer(buffer_size=buffer_size, seed=seed)
    multi_agent = MultiAgent(
        agents=create_agents(),
        replay_buffer=replay_buffer,
        full_action_dim=full_action_dim,
        episodes_before_train=episodes_before_train,
        device=device,
        batch_size=batch_size,
        discount=0.99,
        initial_noise_scale=1.0,
        noise_reduction=0.99,
        seed=seed)
    scores = train(env=env, agent=multi_agent)
    plot_scores(scores)


if __name__ == '__main__':
    run()
