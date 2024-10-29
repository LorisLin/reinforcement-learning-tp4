import numpy as np
import torch
import random
from collections import deque, namedtuple

# Prioritized Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.priorities = deque(maxlen=capacity)
        self.position = 0

    def add(self, transition, error):
        priority = (error + 1e-5) ** self.alpha
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.priorities.append(priority)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta):
        scaled_priorities = np.array(self.priorities) ** beta
        sampling_probs = scaled_priorities / sum(scaled_priorities)
        indices = random.choices(range(len(self.memory)), k=batch_size, weights=sampling_probs)
        
        samples = [self.memory[i] for i in indices]
        weights = (len(self.memory) * sampling_probs[indices]) ** -1
        weights /= max(weights)
        weights = torch.tensor(weights, dtype=torch.float32)
        batch = list(zip(*samples))
        return batch, weights, indices

    def update_priorities(self, indices, errors):
        for i, error in zip(indices, errors):
            self.priorities[i] = (error + 1e-5) ** self.alpha