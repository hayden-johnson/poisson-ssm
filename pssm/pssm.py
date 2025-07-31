from pssm.utils import cumulative_spike_matrix
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import entropy
import matplotlib.pyplot as plt
from tqdm import trange

class PoissonSSM:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder 

    def get_rts(self, img, threshold, n_repeats=10, n_timesteps=100, device='cpu', plot_results=True, tqdm=True):
        all_times, all_actions, all_entropies, all_spikes = [], [], [], []

        dist, _, _, _ = self.encoder.forward(img.unsqueeze(0).unsqueeze(0))
        # encode the image into latent space
        # for some number of repeats
        iterator = trange(n_repeats) if tqdm else range(n_repeats)
        for _ in iterator:
            # sample spikes train
            sample, indicator, times = dist.rsample(hard=True, return_indicator=True)
            # convert to cumulative spike matrix
            cumulative_matrix = cumulative_spike_matrix(indicator, times, n_steps=n_timesteps)
            # decode each column of the cumulative spike matrix
            logits = [self.decoder(x.float().to(device)) for x in cumulative_matrix.T]
            probs = [nn.functional.softmax(logit, dim=-1) for logit in logits]
            entropies = np.array([entropy(prob.detach().cpu().numpy()) for prob in probs])
            
            # find the first index where entropy is below the threshold
            below_threshold = np.where(entropies < threshold)[0]
            if len(below_threshold) > 0:
                time_index = below_threshold[0].item()
            else:
                # If entropy never drops below threshold, use the last timestep
                time_index = len(entropies) - 1
                
            action_index = np.argmax(probs[time_index].detach().cpu().numpy())
            spikes = cumulative_matrix[:, time_index].detach().cpu().numpy().sum()

            # get time, action, and entropy trace
            all_times.append(time_index)
            all_actions.append(action_index)
            all_entropies.append(entropies)
            all_spikes.append(spikes)

        all_times = np.array(all_times)
        all_actions = np.array(all_actions)
        all_entropies = np.array(all_entropies)
        all_spikes = np.array(all_spikes)

        if plot_results:
            plt.imshow(img.reshape(28, 28), cmap='gray')

            fig, axs = plt.subplots(1, 3, figsize=(16, 4))
            axs[0].hist(all_times, bins=30, color='slategray', alpha=0.5)
            axs[0].set_title("Time Index Distribution")
            axs[1].hist(all_actions, bins=30, color='slategray', alpha=0.5)
            axs[1].set_title("Action Index Distribution")
            axs[1].set_xticks(np.arange(10))
            axs[2].set_title("Entropy Distribution")
            for i in range(min(len(all_entropies), 100)):
                axs[2].plot(np.arange(len(all_entropies[i])), all_entropies[i], color='slategray', alpha=0.02)
            axs[2].plot(np.arange(len(all_entropies[0])), np.mean(all_entropies, axis=0), color='slategrey', linewidth=2, label='Mean Entropy')
            # plot line at threshold
            axs[2].axhline(y=threshold, color='black', linestyle='--', label='Threshold', linewidth=.7, alpha=0.7)
            axs[2].legend()
            plt.tight_layout()
            plt.show()

        # return results
        return all_times, all_actions, all_entropies, all_spikes