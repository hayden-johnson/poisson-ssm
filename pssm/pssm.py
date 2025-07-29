from pssm.utils import cumulative_spike_matrix
import torch
import torch.nn as nn
from scipy.stats import entropy

class PoissonSSM:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder 

    def get_rts(self, img, n_repeats=10, n_timesteps=100, device='cpu'):
        dist, _, _, _ = self.encoder.forward(img.unsqueeze(0).unsqueeze(0))
        # encode the image into latent space
        # for some number of repeats
        for _ in range(n_repeats):
            # sample spikes train
            sample, indicator, times = dist.rsample(hard=True, return_indicator=True)
            # convert to cumulative spike matrix
            cumulative_matrix = cumulative_spike_matrix(indicator, times, n_steps=n_timesteps)
            
            print("Cumulative matrix shape:", cumulative_matrix.shape)
            
            
            
            # decode each col into a decoder posterior/entropy
            logits = [self.decoder(x.float().to(device)) for x in cumulative_matrix.T]
            
            print("Logits shape:", logits)
            
            probs = [nn.functional.softmax(logit, dim=-1) for logit in logits]
            entropies = [entropy(prob.detach().cpu().numpy()) for prob in probs]
            
            # decoder posterior/entropy at each timestep
            # get time, action, and entropy trace
        # return results
        return cumulative_matrix, logits, probs, entropies