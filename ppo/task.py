import multiprocessing as mp
# from memory_profiler import profile


class EnvProcess(mp.Process):
    Step = 0
    Reset = 1
    Exit = 2

    def __init__(self, pipe, create_env):
        mp.Process.__init__(self)
        self.pipe = pipe
        self.create_env = create_env

    def run(self):
        env = self.create_env()
        while True:
            op, data = self.pipe.recv()
            if op == EnvProcess.Step:
                # self.step(env, data)
                self.pipe.send(env.step(data))
            elif op == EnvProcess.Reset:
                self.pipe.send(env.reset())
            elif op == EnvProcess.Exit:
                self.pipe.close()
                break
            else:
                raise Exception(f"Unknown command: {op}")

    # @profile
    # def step(self, env, data):
    #     self.pipe.send(env.step(data))


class ProcessTask:
    def __init__(self, create_env):
        sender_pipe, receiver_pipe = mp.Pipe()
        self.pipe = sender_pipe
        self.worker = EnvProcess(receiver_pipe, create_env)
        self.worker.start()

    def step(self, action):
        self.pipe.send([EnvProcess.Step, action])
        return self.pipe.recv()

    def reset(self):
        self.pipe.send([EnvProcess.Reset, None])
        return self.pipe.recv()

    def close(self):
        self.pipe.send([EnvProcess.Exit, None])


class ParallelTask:
    def __init__(self, create_env, n_tasks):
        self.tasks = [ProcessTask(create_env=create_env) for _ in range(n_tasks)]
        self.size = n_tasks

    # @profile
    def step(self, actions):
        results = [task.step(action) for task, action in zip(self.tasks, actions)]
        return results

    def reset(self):
        results = [task.reset() for task in self.tasks]
        return results

    def close(self):
        for task in self.tasks:
            task.close()
