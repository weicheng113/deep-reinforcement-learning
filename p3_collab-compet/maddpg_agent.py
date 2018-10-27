import torch
import numpy as np
import torch.nn.functional as f
import torch.optim as optim


class MultiAgent:
    def __init__(self, num_agents, create_actor, create_critic, replay_buffer, create_noise, state_dim, action_dim,
                 episodes_before_train, device="cpu", lr_actor=1e-4, lr_critic=1e-3, batch_size=128, discount=0.99,
                 tau=1e-3, initial_noise_scale=1.0, noise_reduction=0.999998, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.agents = []
        for _ in range(num_agents):
            noise = create_noise()
            agent = Agent(create_actor=create_actor, create_critic=create_critic, state_dim=state_dim, noise=noise,
                          device=device, lr_actor=lr_actor, lr_critic=lr_critic, tau=tau, seed=seed)
            self.agents.append(agent)

        self.num_agents = num_agents
        self.action_dim = action_dim
        self.whole_action_dim = action_dim * num_agents
        self.buffer = replay_buffer
        self.episodes_before_train = episodes_before_train
        self.device = device
        self.batch_size = batch_size
        self.discount = discount
        self.noise_scale = initial_noise_scale
        self.noise_reduction = noise_reduction
        self.i_episode = 0

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def act(self, states, add_noise=True):
        selected_noise_scale = self.noise_scale
        if not add_noise:
            selected_noise_scale = 0.0
        elif (self.i_episode >= self.episodes_before_train) and (self.noise_scale > 0.01):
            self.noise_scale *= self.noise_reduction
            selected_noise_scale = self.noise_scale

        actions = [agent.act(s, noise_scale=selected_noise_scale) for s, agent in zip(states, self.agents)]
        return np.array(actions)

    def step(self, i_episode, states, actions, rewards, next_states, dones):
        full_state = states.reshape(-1)
        full_next_state = next_states.reshape(-1)
        self.buffer.add(state=states, full_state=full_state, action=actions, reward=rewards,
                        next_state=next_states, full_next_state=full_next_state, done=dones)

        self.i_episode = i_episode
        if (i_episode >= self.episodes_before_train) and (self.buffer.size() >= self.batch_size):
            if (self.i_episode == self.episodes_before_train) and np.any(dones):
                print("\nStart training...")

            for agent_i in range(self.num_agents):
                samples = self.buffer.sample(self.batch_size)
                self.learn(agent_i, self.to_tensor(samples))
            self.soft_update_all()

    def soft_update_all(self):
        for agent in self.agents:
            agent.soft_update_all()

    def to_tensor(self, samples):
        states, full_states, actions, rewards, next_states, full_next_states, dones = samples

        states = torch.from_numpy(states).float().to(self.device)
        full_states = torch.from_numpy(full_states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        full_next_states = torch.from_numpy(full_next_states).float().to(self.device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(self.device)

        return states, full_states, actions, rewards, next_states, full_next_states, dones

    def learn(self, agent_i, samples):
        agent = self.agents[agent_i]
        sampled_states, sampled_full_states, sampled_actions, sampled_rewards, sampled_next_states, \
            sampled_full_next_states, sampled_dones = samples
        agent_rewards = sampled_rewards[:, agent_i].view(-1, 1)
        agent_dones = sampled_dones[:, agent_i].view(-1, 1)

        # Update critic
        next_actions = self.target_act(sampled_next_states)
        q_target_next = agent.critic_target(
            sampled_full_next_states,
            next_actions.view(-1, self.whole_action_dim))
        q_target = agent_rewards + self.discount * q_target_next * (1.0 - agent_dones)
        q_local = agent.critic(sampled_full_states, sampled_actions.view(-1, self.whole_action_dim))

        critic_loss = f.mse_loss(input=q_local, target=q_target.detach())

        agent.critic.zero_grad()
        critic_loss.backward()
        agent.critic_optimizer.step()

        # Update the actor policy
        agent_states = sampled_states[:, agent_i]
        agent_actions = agent.actor(agent_states)
        actions = sampled_actions.clone()
        actions[:, agent_i] = agent_actions

        actor_objective = agent.critic(
            sampled_full_states,
            actions.view(-1, self.whole_action_dim)).mean()
        agent.actor.zero_grad()
        (-actor_objective).backward()
        agent.actor_optimizer.step()

        actor_loss_value = (-actor_objective).cpu().detach().item()
        critic_loss_value = critic_loss.cpu().detach().item()
        return actor_loss_value, critic_loss_value

    def target_act(self, states):
        actions = torch.zeros(states.shape[:2] + (self.action_dim,), dtype=torch.float, device=self.device)
        for i in range(self.num_agents):
            actions[:, i, :] = self.agents[i].actor_target(states[:, i])
        return actions

    def local_act(self, states):
        actions = torch.zeros(states.shape[:2] + (self.action_dim,), dtype=torch.float, device=self.device)
        for i in range(self.num_agents):
            actions[:, i, :] = self.agents[i].actor(states[:, i])
        return actions


class Agent:
    def __init__(self, create_actor, create_critic, state_dim, noise, device, lr_actor, lr_critic, tau=1e-3, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.actor = create_actor().to(device)
        self.actor_target = create_actor().to(device)
        self.actor_optimizer = optim.Adam(params=self.actor.parameters(), lr=lr_actor)

        self.critic = create_critic().to(device)
        self.critic_target = create_critic().to(device)
        self.critic_optimizer = optim.Adam(params=self.critic.parameters(), lr=lr_critic)

        self.state_dim = state_dim
        self.noise = noise
        self.device = device

        self.tau = tau

        Agent.hard_update(model_local=self.actor, model_target=self.actor_target)
        Agent.hard_update(model_local=self.critic, model_target=self.critic_target)

    def act(self, states, noise_scale=0.0):
        states = torch.from_numpy(states).float().to(device=self.device).view(-1, self.state_dim)
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(states).data.numpy()
        self.actor.train()

        actions = self.add_noise2(actions, noise_scale)
        return np.clip(actions.squeeze(), -1, 1)

    def add_noise1(self, actions, noise_scale):
        actions += noise_scale * self.noise.sample()
        return actions

    def add_noise2(self, actions, noise_scale):
        actions += noise_scale * np.random.randn(2)
        return actions

    def reset(self):
        self.noise.reset()

    def soft_update_all(self):
        Agent.soft_update(model_local=self.critic, model_target=self.critic_target, tau=self.tau)
        Agent.soft_update(model_local=self.actor, model_target=self.actor_target, tau=self.tau)

    @staticmethod
    def soft_update(model_local, model_target, tau):
        for local_param, target_param in zip(model_local.parameters(), model_target.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    @staticmethod
    def hard_update(model_local, model_target):
        Agent.soft_update(model_local=model_local, model_target=model_target, tau=1.0)
