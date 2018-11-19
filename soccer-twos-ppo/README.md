[//]: # (Image References)

[image1]: https://github.com/Unity-Technologies/ml-agents/raw/master/docs/images/soccer.png "Soccer Twos"


# Project 3 Challenge: Collaboration and Competition

### Introduction

For this project, you will work with the [Soccer Twos](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#soccer-twos) environment.

### Environment Description
#### States
* The observation space consists of 336 dimensions.
* 3 frames stack together 336 = 3 * 112. Each frame has 112 dimensions. It is corresponding to local 14 ray casts, each detecting 7 possible object types, along with the object's distance. Perception is in 180 degree view from front of agent.

#### Actions(Discrete)
* Striker: 6 actions corresponding to forward, backward, sideways movement, as well as rotation.
* Goalie: 4 actions corresponding to forward, backward, sideways movement.

#### Rewards For Striker
* +1 When ball enters opponent's goal.
*  -0.1 When ball enters own team's goal.
*  -0.001 Existential penalty.

### Rewards For Goalie:
* -1 When ball enters team's goal.
* +0.1 When ball enters opponents goal.
* +0.001 Existential bonus.

#### Goal
* Environment where four agents compete in a 2 vs 2 toy soccer game.
* Striker: Get the ball into the opponent's goal.
* Goalie: Prevent the ball from entering its own goal.
* Solve condition is unknown, but a trained agent should beat a random agent most of time. 

###  Project Structure
The repository contains the following files.
* Soccer.ipynb Contains the agent training code for Unity Tennis environment.
* batcher.py Contains utility class for batching.
* main.py It is an entry file for training in normal python way. It is an alternative to Tennis.ipynb. 
* model.py Contains actor and critic network.
* multi_agent.py Contains MA-PPO based agent implemenation.
* trajectory.py is a utility class for store trajectories.
* soccer_env.py Contains Soccer Twos environment wrapper class.
* train.py Contains training utility methods.
* checkpoints directory are pre-trained model parameter files.
* train.py is a utility class contains training methods.
* test_model.py, test_trajectory.py are unit tests.

### Getting Started
1. Install Anaconda(https://conda.io/docs/user-guide/install/index.html)
2. Install dependencies by issue:
```
pip install -r requirements.txt
```
3. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

4. Place the file in the root folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Soccer.ipynb` to get started with training, 
or directly jump to **Watch Smart Agent** using pre-trained weights, from checkpoints folder, 
to watch the performance of the two trained agents.  
