import math
import torch
import torch.nn as nn

# Poisson distribution class for sampling and KL divergence calculation
class Poisson:
    def __init__(self, log_rate, t=0.0):
        self.log_rate = log_rate
        self.rate = torch.exp(
            self.log_rate.clamp(None, 5)
        ) + 1e-6
        self.n_trials = int(math.ceil(max(self.rate.max().item(),1)*5)) # a large enough number of trials to sample from
        self.t = t

    def rsample(self, hard: bool = False, return_indicator=False):
        x = torch.distributions.Exponential(self.rate).rsample((self.n_trials,))  # inter-event times
        times = torch.cumsum(x, dim=0)  # arrival times of events
        indicator = times < 1.0 # did events arrive before the end of the time interval
        if not (hard or self.t == 0): # soften the indicator function
            indicator = torch.sigmoid((1.0 - times) / self.t)
        if return_indicator:
          return indicator.sum(0).float(), indicator, times
        else:
          return indicator.sum(0).float()
        
    def kl(self, prior, du):
        #"prior" argument referes to log rate of prior
        #equation is r * (1 - dr + dr * log(dr))
        r = torch.exp(prior.clamp(None, 5)) + 1e-6
        rdr = self.rate #final rate is rdr
        logdr = du #log of the modulation of prior rate
        return r-rdr+rdr*logdr

# Minimal Poisson VAE
# TODO: add convolutional layers?
class PVAE(nn.Module):
    def __init__(self):
        super(PVAE, self).__init__()
        self.encode = nn.Sequential(
            nn.Linear(784, 128),
        )
        self.decode = nn.Sequential(
            nn.Linear(128, 784),
            nn.Sigmoid(),
        )
        self.prior = nn.Parameter(torch.zeros((1, 128)))
        self.t = 1.0 #temperature

    def forward(self, x):
        validation = not torch.is_grad_enabled()
        du = self.encode(x).clamp(None, 5)
        dist = Poisson((du + self.prior.clamp(None, 5)).clamp(None, 5), self.t)
        z = dist.rsample(hard=validation)
        y = self.decode(z)
        return dist, du, z, y
    
# Lightning module for training the PVAE
class PL_PVAE(pl.LightningModule):
    def __init__(self):
        super(PL_PVAE, self).__init__()
        self.model = PVAE()
        self.opt = torch.optim.Adam
        self.opt_params = {
            'lr': 1e-3,
        }
        self.beta = 0.0

    def forward(self, x):
        return self.model(x[0].flatten(1))
    
    def training_step(self, batch, batch_idx):
        epoch = self.current_epoch + batch_idx/len(train_dl)
        self.beta = min(5.0, 5*epoch/250)
        self.model.t = max((1.0 - 0.95*epoch/250), 0.05)
        self.log('beta', self.beta)
        self.log('t', self.model.t)

        dist, du, z, y = self(batch)
        kl = dist.kl(self.model.prior, du).mean()
        mse = ((y - batch[0])**2).sum(-1).mean()
        loss = self.beta*kl + mse
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_elbo', (kl + mse).item(), on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        dist, du, z, y = self(batch)
        kl = dist.kl(self.model.prior, du).mean()
        mse = ((y - batch[0])**2).sum(-1).mean()
        loss = self.beta*kl + mse

        self.log('val_mse', mse.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_kl', kl.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_elbo', (kl + mse).item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('l0_sparsity', (z == 0).float().mean().item(), on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return self.opt(self.parameters(), **self.opt_params)