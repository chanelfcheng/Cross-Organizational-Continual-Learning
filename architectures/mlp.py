from torch import nn

class MLP(nn.Module):
    """
    MLP model for network intrusion detection
    """
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.num_in_features = num_features
        self.num_classes = num_classes

        self.layer1 = nn.Linear(num_features, 100)
        self.layer2 = nn.Linear(100, 200)
        self.layer3 = nn.Linear(200, 500)
        self.layer4 = nn.Linear(500, 200)
        self.layer5 = nn.Linear(200, 100)
        self.fc = nn.Linear(100, num_classes)

        self.act = nn.ReLU()
        # self.softmax = nn.Softmax(dim=0)

    def forward(self, x, return_embedding=False):
        x = self.act(self.layer1(x))
        x = self.act(self.layer2(x))
        x = self.act(self.layer3(x))
        x = self.act(self.layer4(x))
        embed = self.act(self.layer5(x))
        out = self.fc(embed)
        # out = self.softmax(out)

        if return_embedding:
            return out, embed
        else:
            return out
