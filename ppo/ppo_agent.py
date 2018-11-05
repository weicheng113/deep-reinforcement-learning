import torch.optim as optim
import torch
import numpy as np
import torch.nn.functional as f
from batcher import Batcher


class Agent:
    def __init__(self, create_actor, create_critic, num_parallels, optimization_epochs=4, batch_size=256,
                 discount=0.99, epsilon=0.1, entropy_weight=0.01, lr=1e-4, device='cpu', seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.actor = create_actor()
        self.critic = create_critic()
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

        self.discount = discount
        self.epsilon = epsilon
        self.entropy_weight = entropy_weight
        self.device = device
        self.optimization_epochs = optimization_epochs
        self.parallel_trajectory = ParallelTrajectory(n=num_parallels)
        self.batch_size = batch_size
        self.num_parallels = num_parallels

    def act(self, states):
        states = torch.tensor(states, dtype=torch.float, device=self.device)
        self.actor.eval()
        with torch.no_grad():
            actions, probs, _ = self.actor(states)
        self.actor.train()

        return actions.cpu().detach().numpy().squeeze(), probs.cpu().detach().numpy().squeeze()

    def step(self, i_episode, states, actions, action_probs, rewards, next_states, dones):
        self.parallel_trajectory.add(
            parallel_states=states,
            parallel_actions=actions,
            parallel_action_probs=action_probs,
            parallel_rewards=rewards,
            parallel_next_states=next_states,
            parallel_dones=dones)

        if np.any(dones):
            states, actions, action_probs, rewards, next_states, dones = self.parallel_trajectory.numpy()
            returns = self.parallel_trajectory.discounted_returns(self.discount)
            states_tensor, actions_tensor, action_probs_tensor, returns_tensor, next_states_tensor = self.to_tensor(
                states=states,
                actions=actions,
                action_probs=action_probs,
                returns=returns,
                next_states=next_states)
            self.learn(
                states=states_tensor,
                actions=actions_tensor,
                action_probs=action_probs_tensor,
                returns=returns_tensor,
                next_states=next_states_tensor)
            del self.parallel_trajectory
            self.parallel_trajectory = ParallelTrajectory(n=self.num_parallels)

    def to_tensor(self, states, actions, action_probs, returns, next_states):
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        action_probs = torch.from_numpy(action_probs).float().to(self.device)
        returns = torch.from_numpy(returns).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)

        state_shape = (-1, ) + states.shape[-3:]
        value_shape = (-1, )

        states = states.view(state_shape)
        actions = actions.view(value_shape)
        action_probs = action_probs.view(value_shape)
        returns = returns.view(value_shape)
        next_states = next_states.view(state_shape)

        return states, actions, action_probs, returns, next_states

    def learn(self, states, actions, action_probs, returns, next_states):
        values = self.values(states)

        advantages = returns - values
        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1.0e-10)
        # print(f"returns[0]: {returns[0]}")
        # print(f"values[0]: {values[0]}")
        # print(f"advantages[0]: {advantages[0]}")
        # print(f"advantages_normalized[0]: {advantages_normalized[0]}")

        policy_losses = []
        value_losses = []
        for _ in range(self.optimization_epochs):
            batcher = Batcher(num_data_points=len(states), batch_size=self.batch_size)
            batcher.shuffle()
            for batch in batcher.batches():
                sampled_advantages = advantages_normalized[batch]
                sampled_states = states[batch]
                sampled_action_probs = action_probs[batch]
                sampled_actions = actions[batch]
                sampled_returns = returns[batch]

                policy_loss, value_loss = self.learn_policy(
                    sampled_advantages=sampled_advantages,
                    sampled_states=sampled_states,
                    sampled_action_probs=sampled_action_probs,
                    sampled_actions=sampled_actions,
                    sampled_returns=sampled_returns)

        policy_losses.append(float(policy_loss.item()))
        value_losses.append(float(value_loss.item()))

        # the clipping parameter reduces as time goes on
        self.epsilon *= 0.999

        # the regulation term also reduces
        # this reduces exploration in later runs
        self.entropy_weight *= 0.995

        return policy_losses, value_losses

    def values(self, states):
        self.critic.eval()
        with torch.no_grad():
            values = self.critic(states)
        self.critic.train()

        return values.cpu().detach().squeeze()

    def learn_policy(self, sampled_states, sampled_actions, sampled_action_probs, sampled_advantages, sampled_returns):
        # actor
        _, new_probs, entropy = self.actor(sampled_states, sampled_actions)
        ratio = new_probs.view(-1) / sampled_action_probs

        ratio_clipped = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)
        objective = torch.min(ratio * sampled_advantages, ratio_clipped * sampled_advantages)

        # entropy = Agent.prob_entropy(old_probs=sampled_action_probs, new_probs=new_probs)
        policy_loss = -torch.mean(objective + self.entropy_weight * entropy)
        # critic
        values = self.critic(sampled_states)
        value_loss = f.mse_loss(input=sampled_returns, target=values.view(-1))

        # optimize
        loss = policy_loss + 0.5 * value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return policy_loss.cpu().detach().numpy().squeeze(), value_loss.cpu().detach().numpy().squeeze()

    # @staticmethod
    # def prob_entropy(old_probs, new_probs):
    #     # include a regularization term
    #     # this steers new_policy towards 0.5
    #     # prevents policy to become exactly 0 or 1 helps exploration
    #     # add in 1.e-10 to avoid log(0) which gives nan
    #     return -(new_probs * torch.log(old_probs + 1.e-10) + (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.e-10))


class ParallelTrajectory:
    def __init__(self, n):
        self.n_parallels = n
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def add(self, parallel_states, parallel_actions, parallel_action_probs,
            parallel_rewards, parallel_next_states, parallel_dones):
        self.states.append(parallel_states)
        self.actions.append(parallel_actions)
        self.action_probs.append(parallel_action_probs)
        self.rewards.append(parallel_rewards)
        self.next_states.append(parallel_next_states)
        self.dones.append(parallel_dones)

    def discounted_returns(self, discount, last_return=None):
        running_return = np.zeros(self.n_parallels, dtype=np.float)
        if last_return:
            running_return = last_return

        n_rewards = len(self.rewards)
        returns = np.zeros((n_rewards, self.n_parallels), dtype=np.float)
        # for i, dones in enumerate(self.dones):
        #     if np.any(dones):
        #         print(f"step {i}, dones: {dones}")
        for i in reversed(range(n_rewards - 1)):
            rewards = np.array(self.rewards[i])
            dones = np.array(self.dones[i]).astype(np.uint8)
            running_return = rewards + discount * (1.0-dones) * running_return
            returns[i, :] = running_return
        return returns

    def numpy(self):
        return (np.array(self.states, dtype=np.float),
                np.array(self.actions, dtype=np.long),
                np.array(self.action_probs, dtype=np.float),
                np.array(self.rewards, dtype=np.float),
                np.array(self.next_states, dtype=np.float),
                np.array(self.dones, dtype=np.bool))

    def clear(self):
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []
