import torch
import torch.nn as nn

class PoissonSSM:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
