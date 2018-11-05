from env_wrapper import PongDeterministic
from model import Actor, Critic
from ppo_agent import Agent
from train import train, plot_scores
from task import ParallelTask
import tracemalloc
import linecache
import os
import torch


def run():
    # tracemalloc.start()
    # snapshot1 = tracemalloc.take_snapshot()

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
    try:
        load_model_if_exists(agent)
        scores = train(envs=envs, agent=agent, episodes=200)
        plot_scores(scores)
    finally:
        print("complete")
        # snapshot2 = tracemalloc.take_snapshot()
        #
        # top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        #
        # print("[ Top 10 differences ]")
        # for stat in top_stats[:10]:
        #     print(stat)
        # snapshot = tracemalloc.take_snapshot()
        # display_top(snapshot)


def load_model_if_exists(agent):
    if os.path.isfile("checkpoint_actor.pth"):
        agent.actor.load_state_dict(torch.load("checkpoint_actor.pth"))
        agent.critic.load_state_dict(torch.load("checkpoint_critic.pth"))


def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


if __name__ == '__main__':
    run()
