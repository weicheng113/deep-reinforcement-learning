from env_wrapper import PongDeterministic
from model import Actor, Critic
from ppo_agent import Agent
from train import train, plot_scores
from task import ParallelTask


def run():
    def create_env():
        return PongDeterministic()
    envs = ParallelTask(create_env=create_env, n_tasks=8)

    env = create_env()
    action_dim = env.action_space.n
    seed = 0

    def create_actor():
        return Actor(action_dim=action_dim, seed=seed)

    def create_critic():
        return Critic(seed=seed)

    optimization_epochs = 4
    discount = 0.95
    epsilon = 0.1
    entropy_weight = 0.01
    learning_rate = 1e-4

    agent = Agent(
        create_actor=create_actor,
        create_critic=create_critic,
        num_parallels=envs.size,
        optimization_epochs=optimization_epochs,
        discount=discount,
        epsilon=epsilon,
        entropy_weight=entropy_weight,
        lr=learning_rate,
        device='cpu',
        seed=seed)

    scores = train(envs=envs, agent=agent)
    plot_scores(scores)


if __name__ == '__main__':
    run()
