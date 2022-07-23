# CNN
class CNN(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=64, kernel_size=6),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.MaxPool1d(kernel_size=(3), strides=2, padding=(3-1)/2),
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=6),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.MaxPool1d(kernel_size=(3), strides=2, padding=(3-1)/2),
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=6),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.MaxPool1d(kernel_size=(3), strides=2, padding=(3-1)/2),
            nn.Flatten(),
            nn.Linear(in_features=1, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=1, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=1, out_features=out_classes),
            nn.Softmax()
        )
    
    def forward(self, x):
        x = self.conv(x)