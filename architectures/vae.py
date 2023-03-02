import torch
from torch import nn

class IDS_VAE(nn.Module):
    def __init__(self, latent_dim, KL_weight):
        super(IDS_VAE, self).__init__()

        self.latent_dim = latent_dim
        self.KL_weight = KL_weight

        self.criterion = nn.MSELoss()

        self.channel_dims = [1, 32, 64]

        #====================[Encoder]====================
        self.encoder = nn.Sequential(
            nn.Conv2d(                                                                #in: N*1*8*8, out: N*dim[1]*4*4
                in_channels=self.channel_dims[0], out_channels=self.channel_dims[1],
                kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(self.channel_dims[1]),                                     #in and out: N*dim[1]*4*4
            nn.LeakyReLU(),                                                           #in and out: N*dim[1]*4*4
            nn.Conv2d(                                                                #in: N*dim[1]*4*4, out: N*dim[2]*2*2
                in_channels=self.channel_dims[1], out_channels=self.channel_dims[2],
                kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(self.channel_dims[2]),                                     #in and out: N*dim[2]*2*2
            nn.LeakyReLU()                                                            #in and out: N*dim[2]*2*2
        )

        #====================[Reparametrize]====================
        self.mu_layer  = nn.Linear(self.channel_dims[2]*2*2, self.latent_dim)         #in: N*1*(dim[2]*4)*1, out: N*1*latent_dim*1
        self.var_layer = nn.Linear(self.channel_dims[2]*2*2, self.latent_dim)         #in: N*1*(dim[2]*4)*1, out: N*1*latent_dim*1

        #====================[Decode]====================
        self.decode_linear_input = nn.Linear(                                         #in: N*1*latent_dim*1, out: N*1*(dim[2]*4)*1
            in_features=self.latent_dim, out_features=self.channel_dims[2]*2*2
        )


        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(                                                       #in: N*dim[2]*2*2, out: N*dim[1]*4*4
                in_channels=self.channel_dims[2], out_channels=self.channel_dims[1],
                kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(self.channel_dims[1]),                        #in and out: N*dim[1]*4*4
            nn.LeakyReLU(),                                                             #in and out: N*dim[1]*4*4
            nn.ConvTranspose2d(                                                       #in: N*dim[1]*4*4, out: N*dim[1]*8*8
                in_channels=self.channel_dims[1], out_channels=self.channel_dims[1],
                kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(self.channel_dims[1]),                        #in and out: N*dim[1]*8*8
            nn.LeakyReLU(),                                                           #in and out: N*dim[1]*8*8
            nn.Conv2d(                                                                #in: N*dim[1]*8*8, out: N*1*8*8
                in_channels=self.channel_dims[1], out_channels=self.channel_dims[0],
                kernel_size=3, padding=1
            ),
            ##nn.Tanh()                                                                 #in and out: N*1*8*8
            nn.Sigmoid()
        )

    def encode(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        return result

    def reparameterize(self, x):
        mu = self.mu_layer(x)
        var = self.var_layer(x)

        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return (eps * std) + mu, mu, var

    def decode(self, x):
        result = self.decode_linear_input(x)
        result = result.view(-1, self.channel_dims[2], 2, 2)
        result = self.decoder(result)
        return result

    def forward(self, x):
        encoded_output = self.encode(x)
        reparameterized_output, mu, var = self.reparameterize(encoded_output)
        decoded_output = self.decode(reparameterized_output)
        return encoded_output, reparameterized_output, mu, var, decoded_output

    def loss(self, output, input, mu, var):
        mse_loss = self.criterion(output, input)

        KL_loss = torch.mean(-0.5 * torch.sum(1 + var - mu**2 - var.exp(), dim=1), dim=0)

        loss = mse_loss + self.KL_weight * KL_loss

        return loss, mse_loss, KL_loss