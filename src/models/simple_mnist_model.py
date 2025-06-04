import torch.nn as nn

class SimpleMnistMode:
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)
        