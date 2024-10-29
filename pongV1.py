import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from DQN import DQN 
from DuelingDQN import DuelingDQN
from PrioritizedReplayBuffer import PrioritizedReplayBuffer
import cv2

import ale_py

gym.register_envs(ale_py)

# Hyperparamètres
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE = 1000
ALPHA = 0.6  # Importance sampling exponent for prioritized replay
BETA = 0.4   # Initial value of beta for prioritized replay
BETA_INCREMENT = 0.001

# Prépocessus d'observation pour réduire la taille des frames
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84))
    return np.array(frame, dtype=np.float32) / 255.0

# Fonction d'entraînement du DQN avec priorité
def train_dqn(agent, target_agent, memory, optimizer, beta):
    if len(memory.memory) < BATCH_SIZE:
        return

    batch, weights, indices = memory.sample(BATCH_SIZE, beta)
    
    # Vérification de la taille des transitions dans `batch`
    try:
        states, actions, rewards, next_states, dones = zip(*batch)
    except ValueError as e:
        print("Erreur de déballage des transitions dans le batch:", e)
        return
    
    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    q_values = agent(states).gather(1, actions).squeeze()
    next_actions = agent(next_states).argmax(1, keepdim=True)
    next_q_values = target_agent(next_states).gather(1, next_actions).squeeze()
    target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

    errors = (q_values - target_q_values.detach()).abs().cpu().numpy()
    loss = (weights * nn.MSELoss(reduction='none')(q_values, target_q_values.detach())).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    memory.update_priorities(indices, errors)
    
# Configuration de l'environnement et de l'agent
env = gym.make("ALE/Pong-v5", render_mode="human")
input_dim = 4
output_dim = env.action_space.n

#agent = DQN(input_dim, output_dim)
#target_agent = DQN(input_dim, output_dim)
agent = DuelingDQN(input_dim, output_dim)
target_agent = DuelingDQN(input_dim, output_dim)

target_agent.load_state_dict(agent.state_dict())
optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

memory = PrioritizedReplayBuffer(MEMORY_SIZE, ALPHA)

epsilon = EPSILON_START
beta = BETA
episode_rewards = []
action_count = 0

for episode in range(1, 501):
    obs, _ = env.reset()
    obs = preprocess_frame(obs)
    state = np.stack([obs] * 4, axis=0)
    episode_reward = 0
    done = False

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action = agent(state_tensor).argmax().item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = preprocess_frame(next_obs)
        next_state = np.concatenate((state[1:], np.expand_dims(next_obs, 0)), axis=0)
        done = terminated or truncated
        episode_reward += reward

        transition = (state, action, reward, next_state, done)
        error = abs(reward)  # Initial error based on reward; can adjust as needed
        memory.add(transition, error)
        state = next_state

        train_dqn(agent, target_agent, memory, optimizer, beta)

        if action_count % TARGET_UPDATE == 0:
            target_agent.load_state_dict(agent.state_dict())

        action_count += 1

    epsilon = max(EPSILON_END, EPSILON_DECAY * epsilon)
    beta = min(1.0, beta + BETA_INCREMENT)
    episode_rewards.append(episode_reward)
    print(f"Episode {episode}, Reward: {episode_reward}, Epsilon: {epsilon}, Beta: {beta}")

env.close()
