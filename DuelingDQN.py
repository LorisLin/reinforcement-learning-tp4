import torch
import torch.nn as nn
import torch.optim as optim

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.fc1_adv = nn.Linear(64 * 7 * 7, 512)
        self.fc1_val = nn.Linear(64 * 7 * 7, 512)
        self.fc2_adv = nn.Linear(512, output_dim)
        self.fc2_val = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        adv = torch.relu(self.fc1_adv(x))
        val = torch.relu(self.fc1_val(x))
        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(-1, adv.size(1))
        
        return val + adv - adv.mean(1, keepdim=True)