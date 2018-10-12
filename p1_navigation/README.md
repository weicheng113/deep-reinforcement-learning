[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

### Environment Description
#### States
* The state space has 37 dimensions.
* Contains the agent's velocity.
* As well as ray-based perception of objects around agent's forward direction.
  
#### Action
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

#### Rewards:
* **+1**: collecting a yellow banana.
* **-1**: for collecting a blue banana.
  
Goal:
* To collect as many yellow bananas as possible while avoiding blue bananas. 
* It is considered solved if the agent get an average score of +13 over 100 consecutive episodes.

### Project Structure
The repository contains the following files:
* network.py Contains simple deep neural network.
* dueling_network.py Contains a network implements Dueling Network from [the paper](https://arxiv.org/abs/1511.06581)
* dqn_agent.py Contains Q-Network agent.
* ddqn_agent.py Contains double Q-Network agent.
* ddqn_prioritized_agent.py Contains double Q-Network agent with prioritized experience replay.
* prioritized_replay_buffer.py Contains prioritized experience replay buffer implementation.
* sum_tree.py Contains a more efficient priority-based sampling structure, the implementation of which references the one from [Jaromir's blog post](https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/).
* Navigation.ipynb Contains the agent training code for Unity Banana environment.
* Report.md Contains the description of the implementation details.
  
### Getting Started
1. Install Anaconda(https://conda.io/docs/user-guide/install/index.html)
2. Install dependencies by issue:
```
pip install -r requirements.txt
```
3. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

4. Place the file in the root folder, and unzip (or decompress) the file. 

### Instructions

Follow the steps in `Navigation.ipynb` to get started with training.

