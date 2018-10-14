import numpy as np


class PongHelper:
    default_background_color = np.array([144, 72, 17])

    @staticmethod
    def preprocess(frame, background_color=default_background_color):
        # image of shape: (210, 160, 3)
        # horizontally (210 - 50) / 2 = 80, vertically 160 / 2 = 80
        processed = np.mean(frame[34:-16:2, ::2] - background_color, axis=-1) / 255.
        # image of shape: (80, 80)
        return processed

    @staticmethod
    def stack_frames(frames, background_color=default_background_color):
        arr = [PongHelper.preprocess(frame, background_color) for frame in frames]
        return np.array(arr)

    @staticmethod
    def frames_to_states(frames_prev, frames):
        return [PongHelper.stack_frames([frame_prev, frame]) for frame_prev, frame in zip(frames_prev, frames)]

    @staticmethod
    def collect_trajectories(envs, agent, t_max, n_warmup=5, action_size=3):
        frames = envs.reset()
        frames_prev = frames
        # randomly keep n frames.
        for _ in range(n_warmup):
            results = envs.step(np.random.randint(action_size, size=envs.size))
            frames_prev = frames
            frames, _, _ = PongHelper.aggregate_results(results)

        n_envs = envs.size
        env_experiences = MultiEnvExperience(n_envs)
        states = PongHelper.frames_to_states(frames_prev, frames)
        for _ in range(t_max):
            actions, probs = agent.act(states)
            results = envs.step(actions)

            frames_prev = frames
            frames, rewards, dones = PongHelper.aggregate_results(results)
            next_states = PongHelper.frames_to_states(frames_prev=frames_prev, frames=frames)
            env_experiences.add(
                probs=probs,
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                dones=dones)

            states = next_states
            if np.any(dones):
                break

        return env_experiences.numpy()

    @staticmethod
    def aggregate_results(results):
        frames = []
        rewards = []
        dones = []

        for result in results:
            frame, reward, done, _ = result
            frames.append(frame)
            rewards.append(reward)
            dones.append(done)

        return frames, rewards, dones


class EnvExperience:
    def __init__(self):
        self.probs = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def add(self, action_prob, state, action, reward, next_state, done):
        self.probs.append(action_prob)
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def numpy(self):
        return (np.array(self.probs, dtype=np.float),
                np.array(self.states, dtype=np.float),
                np.array(self.actions, dtype=np.long),
                np.array(self.rewards, dtype=np.float),
                np.array(self.next_states, dtype=np.float),
                np.array(self.dones, dtype=np.bool))


class MultiEnvExperience:
    def __init__(self, n):
        self.n = n
        self.experiences = []
        for _ in range(n):
            self.experiences.append(EnvExperience())

    def add(self, probs, states, actions, rewards, next_states, dones):
        for i in range(self.n):
            self.experiences[i].add(
                action_prob=probs[i],
                state=states[i],
                action=actions[i],
                reward=rewards[i],
                next_state=next_states[i],
                done=dones[i])

    def numpy(self):
        env_probs = []
        env_states = []
        env_actions = []
        env_rewards = []
        env_next_states = []
        env_dones = []
        for experience in self.experiences:
            probs, states, actions, rewards, next_states, dones = experience.numpy()
            env_probs.append(probs)
            env_states.append(states)
            env_actions.append(actions)
            env_rewards.append(rewards)
            env_next_states.append(next_states)
            env_dones.append(dones)
        return (np.array(env_probs, dtype=np.float),
                np.array(env_states, dtype=np.float),
                np.array(env_actions, dtype=np.long),
                np.array(env_rewards, dtype=np.float),
                np.array(env_next_states, dtype=np.float),
                np.array(env_dones, dtype=np.bool))
