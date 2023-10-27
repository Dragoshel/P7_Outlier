
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        k = 3
                
        self.first_feat_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=k),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=k), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout2d(0.25)
        )
        
        self.second_feat_layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=k),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=k),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout2d(0.25)
        )
        
        self.fully_conn = nn.Sequential(
            nn.Flatten(),
            # 64*2*2*2*2 (A 2 per convolutional layer, 64 is the channels from the last convolutional layer)
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 3),
            # Function for use with multi class clasification
            nn.LogSoftmax(dim=1)
        )
    
    # Progresses data across layers
    def forward(self, x):
        out = self.first_feat_layer(x)
        out = self.second_feat_layer(out)    
        
        out = self.fully_conn(out)
        return out