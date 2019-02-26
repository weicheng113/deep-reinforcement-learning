import torch.optim as optim
import torch
import numpy as np
import torch.nn.functional as f
from batcher import Batcher
from trajectory import ParallelTrajectory
import torch.nn as nn


class PPOAgent:
    def __init__(self, create_actor, create_critic, state_size, num_agents,
                 optimization_epochs, batch_size, discount, clip_ratio, value_loss_weight,
                 entropy_weight, entropy_reduction_rate, lr, max_grad_norm,
                 device, seed, logger):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.actor = create_actor()
        self.critic = create_critic()
        self.actor_critic_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = optim.Adam(self.actor_critic_parameters, lr=lr)

        self.state_size = state_size
        self.discount = discount
        self.clip_ratio = clip_ratio
        self.value_loss_weight = value_loss_weight
        self.entropy_weight = entropy_weight
        self.entropy_reduction_rate = entropy_reduction_rate
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.optimization_epochs = optimization_epochs
        self.parallel_trajectory = ParallelTrajectory(n=num_agents)
        self.batch_size = batch_size
        self.logger = logger

    def act(self, states):
        states = torch.tensor(states, dtype=torch.float, device=self.device).view(-1, self.state_size)
        self.actor.eval()
        with torch.no_grad():
            actions_tensor, probs_tensor, _ = self.actor(states)
        self.actor.train()

        actions = actions_tensor.cpu().detach().numpy().squeeze()
        probs = probs_tensor.cpu().detach().numpy().squeeze()

        if actions.size == 1:
            return actions.item(), probs.item()
        else:
            return actions, probs

    def step(self, states, actions, action_probs, rewards, next_states, dones):
        self.parallel_trajectory.add(
            parallel_states=states,
            parallel_actions=actions,
            parallel_action_probs=action_probs,
            parallel_rewards=rewards,
            parallel_next_states=next_states,
            parallel_dones=dones)

    def episode_done(self, i_episode):
        states, actions, action_probs, rewards, next_states, dones = self.parallel_trajectory.numpy()
        returns = self.parallel_trajectory.discounted_returns(self.discount)
        states_tensor, actions_tensor, action_probs_tensor, returns_tensor, next_states_tensor = self.to_tensor(
            states=states,
            actions=actions,
            action_probs=action_probs,
            returns=returns,
            next_states=next_states)

        policy_objective, value_loss, entropy_value = self.learn(
            states=states_tensor,
            actions=actions_tensor,
            action_probs=action_probs_tensor,
            returns=returns_tensor,
            next_states=next_states_tensor)
        self.logger.add_scalar(f"loss/policy_objective", policy_objective, i_episode)
        self.logger.add_scalar(f"loss/value_loss", value_loss, i_episode)
        self.logger.add_scalar(f"loss/entropy_value", entropy_value, i_episode)
        print(f"policy_objective: {policy_objective}, value_loss: {value_loss}, entropy_value: {entropy_value}")

        self.parallel_trajectory.clear()

    def to_tensor(self, states, actions, action_probs, returns, next_states):
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        action_probs = torch.from_numpy(action_probs).float().to(self.device)
        returns = torch.from_numpy(returns).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)

        states = states.view((-1,) + states.shape[-1:])
        actions = actions.view((-1,) + actions.shape[-1:])
        action_probs = action_probs.view((-1,))
        returns = returns.view((-1,))
        next_states = next_states.view((-1,) + next_states.shape[-1:])

        return states, actions, action_probs, returns, next_states

    def learn(self, states, actions, action_probs, returns, next_states):
        values = self.values(states)

        advantages = returns - values
        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1.0e-10)
        # print(f"returns[0]: {returns[0]}")
        # print(f"values[0]: {values[0]}")
        # print(f"advantages[0]: {advantages[0]}")
        # print(f"advantages_normalized[0]: {advantages_normalized[0]}")

        policy_objectives = []
        value_losses = []
        entropy_values = []
        for _ in range(self.optimization_epochs):
            batcher = Batcher(num_data_points=len(states), batch_size=self.batch_size)
            batcher.shuffle()
            for batch in batcher.batches():
                sampled_advantages = advantages_normalized[batch]
                sampled_states = states[batch]
                sampled_action_probs = action_probs[batch]
                sampled_actions = actions[batch]
                sampled_returns = returns[batch]

                policy_objective, value_loss, entropy_value = self.learn_from_samples(
                    sampled_advantages=sampled_advantages,
                    sampled_states=sampled_states,
                    sampled_action_probs=sampled_action_probs,
                    sampled_actions=sampled_actions,
                    sampled_returns=sampled_returns)

                policy_objectives.append(policy_objective)
                value_losses.append(value_loss)
                entropy_values.append(entropy_value)

        # the clipping parameter reduces as time goes on
        # self.epsilon *= 0.999

        # the regulation term also reduces
        # this reduces exploration in later runs
        self.entropy_weight *= self.entropy_reduction_rate

        return np.mean(policy_objectives), np.mean(value_losses), np.mean(entropy_values)

    def values(self, states):
        self.critic.eval()
        with torch.no_grad():
            values = self.critic(states)
        self.critic.train()
        return values.detach().squeeze()

    def learn_from_samples(self, sampled_states, sampled_actions, sampled_action_probs, sampled_advantages, sampled_returns):
        # actor
        _, new_probs, entropy = self.actor(sampled_states, sampled_actions)
        ratio = (new_probs.view(-1) - sampled_action_probs).exp()

        ratio_clipped = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio)
        objective = torch.min(ratio * sampled_advantages, ratio_clipped * sampled_advantages)

        # entropy = Agent.prob_entropy(old_probs=sampled_action_probs, new_probs=new_probs)
        policy_loss = -torch.mean(objective + self.entropy_weight * entropy)
        # critic
        values = self.critic(sampled_states)
        value_loss = f.mse_loss(input=sampled_returns, target=values.view(-1))

        # optimize
        loss = policy_loss + self.value_loss_weight * value_loss
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.actor_critic_parameters, self.max_grad_norm)
        self.optimizer.step()

        policy_objective_value = objective.mean().cpu().detach().numpy().squeeze().item()
        value_loss_value = self.value_loss_weight * value_loss.cpu().detach().numpy().squeeze().item()
        entropy_value = (self.entropy_weight * entropy).mean().cpu().detach().numpy().squeeze().item()

        return float(policy_objective_value), float(value_loss_value), float(entropy_value)

    def reset(self):
        pass
