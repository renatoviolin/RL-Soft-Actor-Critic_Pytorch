# Deep Reinforcement Learning using PyTorch
Using Soft Actor-Critic (SAC) algorithm from the paper [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)

Implementation adopted from [Chris Yoon](https://github.com/cyoon1729/Policy-Gradient-Methods)

This is a off-policy actor-critic algorithm, and was tested in environment with continous actions.

## Ant

#### Begin of learn: ant dont keep stand up.
<img src="02-ant/img/ant_1.gif" width="300px">

#### After (~100k episodes): ant keep stand up and fall down after a little.
<img src="02-ant/img/ant_2.gif" width="300px">

#### After (~300k episodes): ant can walk without falls.
<img src="02-ant/img/ant_3.gif" width="300px">