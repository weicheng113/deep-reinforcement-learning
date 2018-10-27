from maddpg_agent import MultiAgent
from model import Actor
from model import Critic
from replay_buffer import ReplayBuffer
from noise import OUNoise
import torch
from unity_env_wrapper import EnvMultipleWrapper
from unityagents import UnityEnvironment
from train import train, plot_scores


def run():
    env = UnityEnvironment(file_name="Tennis.app")
    unity_env = env
    env = EnvMultipleWrapper(env=unity_env, train_mode=True)

    buffer_size = int(1e5)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learning_rate_actor = 1e-4
    learning_rate_critic = 1e-3
    seed = 2
    episodes_before_train = 300
    batch_size = 256

    action_dim = env.action_size
    state_dim = env.state_size
    num_agents = env.num_agents

    def create_actor():
        return Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            fc1_units=400,
            fc2_units=300,
            seed=seed)

    def create_critic():
        return Critic(
            state_dim=state_dim * num_agents,
            action_dim=action_dim * num_agents,
            fc1_units=400,
            fc2_units=300,
            seed=seed)

    def create_noise():
        return OUNoise(size=action_dim, seed=seed)

    replay_buffer = ReplayBuffer(buffer_size=buffer_size, seed=seed)
    agent = MultiAgent(
        num_agents=num_agents,
        create_actor=create_actor,
        create_critic=create_critic,
        replay_buffer=replay_buffer,
        create_noise=create_noise,
        state_dim=state_dim,
        action_dim=action_dim,
        episodes_before_train=episodes_before_train,
        device=device,
        lr_actor=learning_rate_actor,
        lr_critic=learning_rate_critic,
        batch_size=batch_size,
        discount=0.99,
        tau=1e-3,
        noise_reduction=0.99,
        seed=seed)
    scores = train(env=env, agent=agent)
    plot_scores(scores)


if __name__ == '__main__':
    run()
