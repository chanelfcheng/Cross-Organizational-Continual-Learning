import torch
from torch import nn

d = 10  # latent dimension, may be changed later

class VAE(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.num_features = num_features

        self.encoder = nn.Sequential(
            nn.Linear(num_features, d ** 2),
            nn.ReLU(),
            nn.Linear(d ** 2, d * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(d, d ** 2),
            nn.ReLU(),
            nn.Linear(d ** 2, num_features),
            nn.Sigmoid(),
        )

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new_empty(std.size()).normal_()
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu_logvar = self.encoder(x.view(-1, self.num_features)).view(-1, 2, d)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar

    def encoder_loss(self, x_hat, x, mu, logvar, w=1):
        """
        loss function for the VAE
        params:
            x_hat: reconstructed input
            x: original input
            mu: mean of the latent distribution
            logvar: log variance of the latent distribution
            w: weight of the KL divergence term
        """
        # print(f'x_hat: {x_hat}, x: {x}, mu: {mu}, logvar: {logvar}')
        BCE = nn.functional.mse_loss(
            x_hat, x.view(-1, self.num_features)
        )
        KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))

        # print(f'BCE: {BCE}, KLD: {KLD}')

        return BCE + w * KLD