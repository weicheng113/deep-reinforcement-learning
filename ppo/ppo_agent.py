import torch.optim as optim
import torch
import numpy as np
import torch.nn.functional as F


class Agent:
    def __init__(self, create_actor, create_critic, optimization_epochs=4, discount=0.995,
                 epsilon=0.1, beta=0.01, actor_lr=1e-4, critic_lr=1e-4, device='cpu'):
        self.actor = create_actor()
        self.critic = create_critic()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=actor_lr)

        self.discount = discount
        self.epsilon = epsilon
        self.beta = beta
        self.device = device
        self.optimization_epochs = optimization_epochs

    def act(self, states):
        states = torch.tensor(states, dtype=torch.float, device=self.device)
        self.actor.eval()
        with torch.no_grad():
            actions, probs = self.actor(states)
        self.actor.train()

        return actions.cpu().detach().numpy().squeeze(), probs.cpu().detach().numpy().squeeze()

    def learn(self, action_probs, states, actions, rewards, next_states, dones):
        value_shape = -1
        state_shape = (-1, ) + states.shape[-3:]

        action_probs = action_probs.reshape(value_shape)
        states = states.reshape(state_shape)
        actions = actions.reshape(value_shape)
        values = self.sampled_values(states).reshape(value_shape)
        returns = np.array([self.discounted_returns(r, d) for r, d in zip(rewards, dones)])
        returns = returns.reshape(value_shape)
        # next_states = next_states.reshape(state_shape)
        # dones = dones.reshape(value_shape)

        advantages = returns - values
        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1.0e-10)

        objectives = []
        losses = []
        for _ in range(self.optimization_epochs):
            objective = self.learn_policy(
                sampled_probs=action_probs,
                sampled_advantages=advantages_normalized,
                sampled_states=states,
                sampled_actions=actions)
            loss = self.learn_value(states=states, sampled_returns=returns)
            objectives.append(objective.item())
            losses.append(loss.item())

        # the clipping parameter reduces as time goes on
        self.epsilon *= 0.999

        # the regulation term also reduces
        # this reduces exploration in later runs
        self.beta *= 0.995

        return objectives, losses

    def sampled_values(self, states):
        states = torch.tensor(states, dtype=torch.float, device=self.device)
        self.critic.eval()
        with torch.no_grad():
            values = self.critic(states)
        self.critic.train()

        return values.cpu().detach().numpy().squeeze()

    def discounted_returns(self, rewards, dones):
        running_return = 0
        returns = np.zeros_like(rewards)
        for i in reversed(range(len(rewards) - 1)):
            running_return = rewards[i] + self.discount * (1-dones[i]) * running_return
            returns[i] = running_return
        return returns

    def learn_policy(self, sampled_probs, sampled_advantages, sampled_states, sampled_actions):
        sampled_probs = torch.tensor(sampled_probs, dtype=torch.float, device=self.device)
        sampled_advantages = torch.tensor(sampled_advantages, dtype=torch.float, device=self.device)
        sampled_states = torch.tensor(sampled_states, dtype=torch.float, device=self.device)
        sampled_actions = torch.tensor(sampled_actions, dtype=torch.long, device=self.device)

        _, probs_new = self.actor(sampled_states, sampled_actions)
        ratio = probs_new / sampled_probs

        ratio_clipped = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)
        objective = torch.min(ratio * sampled_advantages, ratio_clipped * sampled_advantages)

        entropy = Agent.prob_entropy(old_probs=sampled_probs, new_probs=probs_new)

        objective = torch.mean(objective + self.beta*entropy)

        self.actor_optimizer.zero_grad()
        (-objective).backward()
        self.actor_optimizer.step()

        return objective.cpu().detach().numpy().squeeze()

    @staticmethod
    def prob_entropy(old_probs, new_probs):
        # include a regularization term
        # this steers new_policy towards 0.5
        # prevents policy to become exactly 0 or 1 helps exploration
        # add in 1.e-10 to avoid log(0) which gives nan
        return -(new_probs * torch.log(old_probs + 1.e-10) + (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.e-10))

    def learn_value(self, states, sampled_returns):
        states = torch.tensor(states, dtype=torch.float, device=self.device)
        values = self.critic(states)
        sampled_returns = torch.tensor(sampled_returns, dtype=torch.float, device=self.device)

        loss = F.mse_loss(input=sampled_returns, target=values)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return loss.cpu().detach().numpy().squeeze()
