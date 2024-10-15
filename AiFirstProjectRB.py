import numpy as np

import pandas as pd

import gym

import random

env=gym.make("FrozlsenLake-v1")

action_size=env.action_space.n
state_size=env.observation_space.n

qtable=np.zeros((state_size,action_size))
print("Q-table shape",qtable.shape)
print(qtable)

total_episodes=15000
learning_rate=0.8
max_steps=99
gamma=0.95

epsilon=1.0
epsilon_decay_rate=0.0005
max_epsilon=1.0
min_epsilon=0.01
decay_rate=0.005

rewards=[]

for episode in range(total_episodes):
  state=env.reset()
  step=0
  done=False
  total_rewards=0

  for step in range(max_steps):
    exp_exp_tradeoff=random.uniform(0,1)

    if exp_exp_tradeoff>epsilon:
      action=np.argmax(qtable[state,:])
    else:
      action=env.action_space.sample()
    new_state,reward,done,info=env.step(action)

    qtable[state,action]=qtable[state,action]+learning_rate*(reward+gamma*np.max(qtable[new_state,:])-qtable[state,action])
    total_rewards=total_rewards+reward
    state=new_state

    if done==True:
      break
  epsilon=min_epsilon+(max_epsilon-min_epsilon)*np.exp(-decay_rate*episode)
  rewards.append(total_rewards)

  print("Score",str(sum(rewards)/total_episodes))
print("Q-table")
print(qtable)