import numpy as np
import gym
import random
from sklearn.preprocessing import KBinsDiscretizer

env = gym.make('CartPole-v1')
action_space_size = env.action_space.n
state_space_size = [10, 10, 10, 10]  # Define the size of the discrete state space

qtable = np.zeros(state_space_size + [action_space_size])

# Hyperparameters
total_episodes = 10000
learning_rate = 0.2
max_steps = 100
gamma = 0.99

epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay_rate = 0.001
rewards = []

# Discretize the state space
est = KBinsDiscretizer(n_bins=state_space_size, encode='ordinal', strategy='uniform')
est.fit([env.observation_space.low, env.observation_space.high])

def discretize_state(state):
    return tuple(map(int, est.transform([state])[0]))

for episode in range(total_episodes):
    state = discretize_state(env.reset())
    total_rewards = 0
    
    for step in range(max_steps):
        if random.uniform(0, 1) > epsilon:
            action = np.argmax(qtable[state])  # Exploit
        else:
            action = env.action_space.sample()  # Explore
            
        new_state, reward, done, _ = env.step(action)
        new_state = discretize_state(new_state)
        
        max_new_state = np.max(qtable[new_state])
        
        qtable[state + (action,)] = qtable[state + (action,)] + learning_rate * (reward + gamma * max_new_state - qtable[state + (action,)])
        
        total_rewards += reward
        
        state = new_state
        if done:
            break
        
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)
        
print("Average Score:", str(sum(rewards) / total_episodes))
print(qtable)

# Testing the trained agent
for episode in range(5):
    state = discretize_state(env.reset())
    
    print("Episode:", episode+1)
    
    for step in range(max_steps):
        action = np.argmax(qtable[state])
        new_state, reward, done, _ = env.step(action)
        env.render()
        
        if done:
            print("Number of Steps:", step)
            break
        
        state = discretize_state(new_state)
        
env.close()
