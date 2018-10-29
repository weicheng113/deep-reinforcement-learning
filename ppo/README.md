### Project Structure
The repository contains the following files:
* pong-PPO.ipynb Contains the training code. This is the entry file for jupyter note.
* model.py Contains actor and critic network.
* ppo_agent.py Implement Proximal Policy Optimization agent.
* task.py Contains parallel task utility for running multiple same environments at the same time.
* env_wrapper.py Contains the wrapper for the pong environment.
* batcher.py Contains batching utility class.
* train.py Contains training method used by pong-PPO.ipynb and main.py
* main.py Contains app entry code. It can be used to start training without using jupyter notebook.
* test_model.py Unit tests for model.py.
* test_ppo_agent.py Unit tests for ppo_agent.py.
