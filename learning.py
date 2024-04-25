import pandas as pd
import numpy as np
from env import DroidRLEnvironment
import tensorflow as tf

df = pd.read_csv('drebin-215-dataset-5560malware-9476-benign.csv', low_memory=False)
env = DroidRLEnvironment(df,15)
# PARAMETERS
NUM_EPISODES = 5

#Thử random agent
def random_agent():
    return tf.random.uniform(shape=(), minval=0, maxval=df.shape[1] - 1, dtype=tf.int32).numpy()


for EPISODE in range(NUM_EPISODES):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = random_agent()
        while action in state:
            action = random_agent()
        state,reward,done = env.step(action)
        print(state, reward)
        total_reward += reward
    print("Total Reward: ", total_reward)





