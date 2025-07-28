import tqdm
import torch
from torch.utils.data import DataLoader

from pvae.data import cumulative_spike_matrix

def create_decoder_dataset(model, dataset, n_timesteps=100, n_repeats=10, max_samples=1000, device='cpu'):
    """
    Create data of encoder spike observations for decoder training.
    """
    model = model.to(device)
    model.eval()
    
    # Limit dataset size for now
    limited_dataset = torch.utils.data.Subset(dataset, range(min(max_samples, len(dataset))))
    
    # Use DataLoader for better batching
    fast_loader = DataLoader(limited_dataset, batch_size=100, shuffle=False, num_workers=0)
    
    all_data = []
    all_labels = []
    
    with torch.no_grad():
        for batch_imgs, batch_labels in tqdm(fast_loader, desc="Batch"):
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)
            
            # Process each image in the batch
            for img, label in zip(batch_imgs, batch_labels):
                # Encode img into latent space
                dist, du, z, y = model.forward(img.unsqueeze(0).unsqueeze(0))
                
                # Sample multiple times
                for _ in range(n_repeats):
                    sample, indicator, times = dist.rsample(hard=True, return_indicator=True)
                    cumulative_matrix = cumulative_spike_matrix(indicator, times, n_steps=n_timesteps)

                    # Add all timesteps at once
                    all_data.append(cumulative_matrix.T.cpu())  # (n_timesteps, n_neurons)
                    all_labels.extend([label.cpu().item()] * n_timesteps)
    
    # Stack everything
    decoder_data = torch.cat(all_data, dim=0)
    decoder_labels = torch.tensor(all_labels, dtype=torch.long)
    
    return decoder_data, decoder_labels