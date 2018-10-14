### Project Structure
The repository contains the following files:
* my-pong-PPO.ipynb Contains the training code. This is the entry file for jupyter note.
* model.py Contains actor and critic network.
* ppo_agent.py Implement Proximal Policy Optimization agent.
* task.py Contains parallel task utility for running multiple same environments at the same time.
* pong_wrapper.py Contains the wrapper for the pong environment.
* pong_helper.py Contains some helper methods for handling the pong environment, e.g., stacking frames. 
* test_model.py Unit tests for model.py.
* test_pong_helper.py Unit tests for pong_helper.py.
* test_ppo_agent.py Unit tests for ppo_agent.py.
* Other files, which are from the course, can be ignored.
