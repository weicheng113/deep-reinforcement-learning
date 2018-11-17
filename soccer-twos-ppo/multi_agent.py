import torch
import numpy as np
import torch.nn.functional as f
import torch.optim as optim
from batcher import Batcher
from trajectory import ParallelTrajectory


class MultiAgent:
    def __init__(self, agents, device="cpu", discount=0.99, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.agents = agents
        self.device = device
        self.discount = discount
        self.num_agents = len(agents)
        self.parallel_trajectory = ParallelTrajectory(n=self.num_agents)

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def act(self, states):
        action_shape = states.shape[0]
        actions = np.zeros(action_shape, dtype=np.long)
        action_probs = np.zeros(action_shape, dtype=np.float)
        for i in range(self.num_agents):
            state, agent = states[i], self.agents[i]
            action, action_prob = agent.act(state)
            actions[i] = action
            action_probs[i] = action_prob
        return actions, action_probs

    def step(self, states, actions, action_probs, rewards, next_states, dones):
        self.parallel_trajectory.add(
            parallel_states=states,
            parallel_actions=actions,
            parallel_action_probs=action_probs,
            parallel_rewards=rewards,
            parallel_next_states=next_states,
            parallel_dones=dones)

    def episode_done(self, i_episode):
        if True:
        # if i_episode % 1 == 0:
            states, full_states, actions, action_probs, rewards, next_states, dones = self.parallel_trajectory.numpy()
            returns = self.parallel_trajectory.discounted_returns(self.discount)
            states_tensor, full_states_tensor, actions_tensor, action_probs_tensor, returns_tensor, next_states_tensor = \
                self.to_tensor(
                    states=states,
                    full_states=full_states,
                    actions=actions,
                    action_probs=action_probs,
                    returns=returns,
                    next_states=next_states)

            policy_losses = []
            value_losses = []
            for i, agent in enumerate(self.agents):
                policy_loss, value_loss = agent.learn(
                    states=states_tensor[:, i],
                    full_states=full_states_tensor,
                    actions=actions_tensor[:, i],
                    action_probs=action_probs_tensor[:, i],
                    returns=returns_tensor[:, i],
                    next_states=next_states_tensor[:, i])
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
            print(f"policy_losses: {policy_losses}, value_losses: {value_losses}")
            self.parallel_trajectory.clear()

    def to_tensor(self, states, full_states, actions, action_probs, returns, next_states):
        states = torch.from_numpy(states).float().to(self.device)
        full_states = torch.from_numpy(full_states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        action_probs = torch.from_numpy(action_probs).float().to(self.device)
        returns = torch.from_numpy(returns).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)

        return states, full_states, actions, action_probs, returns, next_states


class PPOAgent:
    def __init__(self, create_actor, create_critic, state_dim, optimization_epochs=4, batch_size=256,
                 epsilon=0.1, entropy_weight=0.01, lr=1e-4, device='cpu', seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.actor = create_actor()
        self.critic = create_critic()
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

        self.state_dim = state_dim
        self.epsilon = epsilon
        self.entropy_weight = entropy_weight
        self.device = device
        self.optimization_epochs = optimization_epochs
        self.batch_size = batch_size

    def act(self, states):
        states = torch.tensor(states, dtype=torch.float, device=self.device).view(-1, self.state_dim)
        self.actor.eval()
        with torch.no_grad():
            actions, probs, _ = self.actor(states)
            # actions1, prob1, _ = self.actor(states, actions)
        self.actor.train()

        return actions.cpu().detach().numpy().squeeze(), probs.cpu().detach().numpy().squeeze()

    def learn(self, states, full_states, actions, action_probs, returns, next_states):
        values = self.critic_values(full_states)

        advantages = returns - values
        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1.0e-10)
        # print(f"returns[0]: {returns[0]}")
        # print(f"values[0]: {values[0]}")
        # print(f"advantages[0]: {advantages[0]}")
        # print(f"advantages_normalized[0]: {advantages_normalized[0]}")

        policy_losses = []
        value_losses = []
        policy_loss, value_loss = self.learn_from_samples(
            sampled_advantages=advantages_normalized,
            sampled_states=states,
            sampled_full_states=full_states,
            sampled_action_probs=action_probs,
            sampled_actions=actions,
            sampled_returns=returns)
        # for _ in range(self.optimization_epochs):
        #     batcher = Batcher(num_data_points=len(states), batch_size=self.batch_size)
        #     batcher.shuffle()
        #     for batch in batcher.batches():
        #         sampled_advantages = advantages_normalized[batch]
        #         sampled_states = states[batch]
        #         sampled_full_states = full_states[batch]
        #         sampled_action_probs = action_probs[batch]
        #         sampled_actions = actions[batch]
        #         sampled_returns = returns[batch]
        #
        #         policy_loss, value_loss = self.learn_from_samples(
        #             sampled_advantages=sampled_advantages,
        #             sampled_states=sampled_states,
        #             sampled_full_states=sampled_full_states,
        #             sampled_action_probs=sampled_action_probs,
        #             sampled_actions=sampled_actions,
        #             sampled_returns=sampled_returns)
        #
        #         policy_losses.append(policy_loss)
        #         value_losses.append(value_loss)

        # the clipping parameter reduces as time goes on
        self.epsilon *= 0.999

        # the regulation term also reduces
        # this reduces exploration in later runs
        self.entropy_weight *= 0.995

        return np.mean(policy_losses), np.mean(value_losses)

    def critic_values(self, full_states):
        self.critic.eval()
        with torch.no_grad():
            values = self.critic(full_states)
        self.critic.train()

        return values.cpu().detach().squeeze()

    def learn_from_samples(self, sampled_states, sampled_full_states, sampled_actions,
                           sampled_action_probs, sampled_advantages, sampled_returns):
        # actor
        _, new_probs, entropy = self.actor(sampled_states, sampled_actions)
        ratio = new_probs.view(-1) / sampled_action_probs

        ratio_clipped = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)
        objective = torch.min(ratio * sampled_advantages, ratio_clipped * sampled_advantages)

        # entropy = Agent.prob_entropy(old_probs=sampled_action_probs, new_probs=new_probs)
        policy_loss = -torch.mean(objective + self.entropy_weight * entropy)
        # critic
        values = self.critic(sampled_full_states)
        value_loss = f.mse_loss(input=sampled_returns, target=values.view(-1))

        # optimize
        loss = policy_loss + 0.5 * value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        policy_loss_value = policy_loss.cpu().detach().numpy().squeeze().item()
        value_loss_value = value_loss.cpu().detach().numpy().squeeze().item()
        if np.isnan(policy_loss_value) or np.isnan(value_loss_value):
            print(f"policy_loss_value: {policy_loss_value}, value_loss_value: {value_loss_value}")
        return float(policy_loss_value), float(value_loss_value)

    def reset(self):
        pass
