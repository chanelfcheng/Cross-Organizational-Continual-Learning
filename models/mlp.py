from torch import nn

class MLP(nn.Module):
    """
    MLP model for network intrusion detection
    """
    def __init__(self, num_features, num_classes, embeddings=False):
        super().__init__()

        self.embeddings = embeddings

        self.num_out_features = 100
        self.layer1 = nn.Linear(num_features, 100)
        self.layer2 = nn.Linear(100, 200)
        self.layer3 = nn.Linear(200, 500)
        self.layer4 = nn.Linear(500, 200)
        self.layer5 = nn.Linear(200, self.num_out_features)
        self.fc = nn.Linear(100, num_classes)

        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.act(self.layer1(x))
        x = self.act(self.layer2(x))
        x = self.act(self.layer3(x))
        x = self.act(self.layer4(x))
        features = self.act(self.layer5(x))
        x = self.fc(features)
        x = self.softmax(x)

        if self.embeddings:
            return x, features
        else:
            return x
