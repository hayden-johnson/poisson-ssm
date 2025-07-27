import tqdm
import torch
from torch.utils.data import DataLoader

def create_decoder_dataset(model, dataset, n_timesteps=100, n_repeats=10, max_samples=10000, device='cpu'):
    """
    Ultra-fast version that limits the dataset size and uses maximum vectorization.
    """
    model = model.to(device)
    model.eval()
    
    # Limit dataset size for speed
    limited_dataset = torch.utils.data.Subset(dataset, range(min(max_samples, len(dataset))))
    
    # Use DataLoader for better batching
    fast_loader = DataLoader(limited_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    all_data = []
    all_labels = []
    
    with torch.no_grad():
        for batch_imgs, batch_labels in tqdm(fast_loader, desc="Ultra-fast processing"):
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)
            
            # Process each image in the batch
            for img, label in zip(batch_imgs, batch_labels):
                # Get distribution
                dist, du, z, y = model.forward(img.unsqueeze(0).unsqueeze(0))
                
                # Sample multiple times efficiently
                for _ in range(n_repeats):
                    sample, indicator, times = dist.rsample(hard=True, return_indicator=True)
                    cum_matrix = cumulative_spike_matrix(indicator, times, n_steps=n_timesteps)
                    
                    # Add all timesteps at once
                    all_data.append(cum_matrix.T.cpu())  # (n_timesteps, n_neurons)
                    all_labels.extend([label.cpu().item()] * n_timesteps)
    
    # Stack everything
    decoder_data = torch.cat(all_data, dim=0)
    decoder_labels = torch.tensor(all_labels, dtype=torch.long)
    
    return decoder_data, decoder_labels