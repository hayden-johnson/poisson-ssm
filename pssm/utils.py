import tqdm
import torch

def cumulative_spike_matrix(indicator, times, n_steps=100):
    """
    Create a cumulative spike matrix (n_neurons x n_timesteps) showing 
    cumulative spike counts for each neuron up to each time step.

    Args:
        indicator: boolean tensor of shape (n_trials, batch_size, n_neurons)
        times: tensor of spike times of shape (n_trials, batch_size, n_neurons)  
        n_steps: number of time bins to discretize [0,1] interval

    Returns:
        cumulative_counts: tensor of shape (n_neurons, n_steps) with cumulative spike counts
    """
    # Squeeze out batch dimension (assuming batch_size=1)
    indicator_2d = indicator[:, 0, :]  # (n_trials, n_neurons)
    times_2d = times[:, 0, :]          # (n_trials, n_neurons)

    n_neurons = indicator_2d.shape[1]
    device = times.device

    # Create time bin edges
    edges = torch.linspace(0.0, 1.0, n_steps + 1, device=device)

    # Get valid spike times and corresponding neuron indices
    valid_mask = indicator_2d
    trials, neurons = torch.nonzero(valid_mask, as_tuple=True)
    valid_times = times_2d[trials, neurons]  # Flat array of all spike times

    # Find which time bin each spike falls into
    bin_indices = torch.bucketize(valid_times, edges) - 1
    # Clamp to valid range [0, n_steps-1]
    bin_indices = torch.clamp(bin_indices, 0, n_steps - 1)

    # Create a spike count matrix (n_neurons x n_steps)
    spike_counts = torch.zeros(n_neurons, n_steps, device=device, dtype=torch.int32)

    # Count spikes in each bin for each neuron
    for i in range(len(valid_times)):
        neuron_idx = neurons[i]
        time_bin = bin_indices[i]
        spike_counts[neuron_idx, time_bin] += 1

    # Convert to cumulative counts along time dimension
    cumulative_counts = torch.cumsum(spike_counts, dim=1)

    return cumulative_counts


