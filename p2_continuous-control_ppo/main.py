import torch
from train import train, create_agent
import os
from tensorboardX import SummaryWriter
from unityagents import UnityEnvironment
from unity_env_wrapper import EnvMultipleWrapper


def run_training():
    checkpoints_dir = "checkpoints"
    make_checkpoints_dir(checkpoints_dir)

    # unity_env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64')
    unity_env = UnityEnvironment(file_name='Reacher_Linux_NoVis/Reacher.x86_64')
    env = EnvMultipleWrapper(env=unity_env, train_mode=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    seed = 2
    learning_rate = 1e-4
    batch_size = 256
    clip_ratio = 0.2
    optimization_epochs = 10
    entropy_weight = 0.01
    entropy_reduction_rate = 0.99
    discount = 0.95
    max_grad_norm = 0.5
    value_loss_weight = 0.5
    episodes = 500
    max_t = 1000
    logger = SummaryWriter()
    logger.add_text(
        "Hyper-parameters",
        f"learning_rate: {learning_rate}, batch_size: {batch_size}"
        + f", clip_ratio: {clip_ratio}, optimization_epochs: {optimization_epochs}"
        + f", entropy_weight: {entropy_weight}, entropy_reduction_rate: {entropy_reduction_rate}"
        + f", discount: {discount}, max_grad_norm: {max_grad_norm}, value_loss_weight: {value_loss_weight}"
        + f", episodes: {episodes}, max_t: {max_t}"
        + f", number of agents: {env.num_agents}"
        + f", with 2 batchNorm in critic"
        + f", hidden_units [300, 300] with elu")

    agent = create_agent(
        env=env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        discount=discount,
        clip_ratio=clip_ratio,
        optimization_epochs=optimization_epochs,
        value_loss_weight=value_loss_weight,
        entropy_weight=entropy_weight,
        entropy_reduction_rate=entropy_reduction_rate,
        max_grad_norm=max_grad_norm,
        device=device,
        seed=seed,
        logger=logger)

    train(env=env,
          agent=agent,
          episodes=episodes,
          max_t=max_t,
          print_every=50,
          logger=logger,
          checkpoints_dir=checkpoints_dir)

    env.close()


def make_checkpoints_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    run_training()
