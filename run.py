#!/usr/bin/env python3
"""
Main entry point for training and evaluating PSSM models.
Run this from the project root directory.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import lightning as pl
from torchvision import datasets
from torch.utils.data import DataLoader


# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pssm.encoder import PL_PVAE
from pssm.decoder import MLP
from pssm.data import create_decoder_dataset
from pssm.pssm import PoissonSSM

# Build function to set up the encoder, decoder, and datasets
def build():
    ## SET UP DIRECTORIES
    name = "pssm"
    root_dir = "./data"
    data_dir = f"{root_dir}/Datasets"
    root_dir = f"{root_dir}/{name}"

    load_encoder = True
    encoder_dir = "./networks/encoder.ckpt"

    load_decoder = True
    load_decoder_path = "./networks/decoder.ckpt"

    load_decoder_dataset = True
    decoder_dataset_path = "./data/Datasets/decoder/decoder_dataset_100_5_1000_100.pt"

    # make new checkpoint directory with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = f"{root_dir}/{name}/{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    ## SET UP DEVICE
    train_device = "0"
    device = train_device + "," # lightning device formatting

    ## GET DATASET
    print("Generating encoder dataset...", end=' ')
    bsize=256
    ds_train = datasets.MNIST(data_dir, train=True, download=True).data
    ds_test = datasets.MNIST(data_dir, train=False, download=True).data
    ds_train_targets = datasets.MNIST(data_dir, train=True, download=True).targets
    ds_test_targets = datasets.MNIST(data_dir, train=False, download=True).targets

    ds_train = (ds_train.float().reshape(-1, 784)/255)
    ds_test = (ds_test.float().reshape(-1, 784)/255)
    ds_train = torch.utils.data.TensorDataset(ds_train, ds_train_targets)
    ds_test = torch.utils.data.TensorDataset(ds_test, ds_test_targets)

    # Add num_workers for better performance
    train_dl = DataLoader(ds_train, batch_size=bsize, shuffle=True, num_workers=4)
    val_dl = DataLoader(ds_test, batch_size=bsize, shuffle=False, num_workers=4)
    print("Done.")

    ## SET UP ENCODER
    if load_encoder:
        print("Loading encoder from checkpoint...", end=' ')
        try:
            encoder = PL_PVAE.load_from_checkpoint(encoder_dir)
            print("Done.")
        except Exception as e:
            print(f"Failed to load encoder from checkpoint: {e}")
            sys.exit(1)
    else:
        print("Training encoder...", end=' ')
        encoder = PL_PVAE(len(train_dl))
        trainer_args = {
            "callbacks": [pl.pytorch.callbacks.ModelCheckpoint(dirpath=checkpoint_dir, monitor='val_elbo', save_top_k=1, mode='min', verbose=True)],
            "accelerator": "cpu" if device == 'cpu' else "gpu",
            "gradient_clip_val": 1.0,
        }
        if device != 'cpu':
            trainer_args["devices"] = device
        trainer = pl.Trainer(**trainer_args, default_root_dir=checkpoint_dir, max_epochs=55, num_sanity_val_steps=0)
        trainer.fit(encoder, val_dataloaders=val_dl, train_dataloaders=train_dl)
        print("Done.")

    ## SET UP DECODER
    if load_decoder:
        print("Loading decoder from checkpoint...", end=' ')
        try:
            decoder = MLP(input_size=128, output_size=10)
            decoder.load_state_dict(torch.load(load_decoder_path))
            print("Done.")
        except Exception as e:
            print(f"Failed to load decoder from checkpoint: {e}")
            sys.exit(1)

    else:
        if load_decoder_dataset:
            print("Loading decoder dataset from file...", end=' ')
            # load decoder dataset
            try:
                decoder_ds = torch.load(decoder_dataset_path)
                print("Done.")
            except Exception as e:
                print(f"Failed to load decoder dataset: {e}")
                sys.exit(1)
            #raise NotImplementedError("Loading decoder dataset from file is not implemented yet.")
        else:
            n_timesteps = 100
            n_repeats = 5
            max_samples = 1000
            batch_size = 100
            print("Creating decoder dataset...", end=' ')
            device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
            # create decoder dataset with encoder model
            decoder_data, decoder_labels = create_decoder_dataset(encoder, ds_train, n_timesteps=n_timesteps, n_repeats=n_repeats, max_samples=max_samples, batch_size=batch_size, device=device)
            # create dataload for decoder
            decoder_ds = torch.utils.data.TensorDataset(decoder_data, decoder_labels)
            print("Done.")
            # save out dataset
            try:
                print("Saving decoder dataset...", end=' ')
                torch.save(decoder_ds, f"{checkpoint_dir}/decoder_dataset_{n_timesteps}_{n_repeats}_{max_samples}_{batch_size}.pt")
                print("Done.")
            except Exception as e:
                print(f"Failed to save decoder dataset: {e}")

        decoder_dl = DataLoader(decoder_ds, batch_size=bsize, shuffle=True, num_workers=4)

        print("Training decoder...", end=' ')
        input_size = 128    # Number of neurons
        output_size = 10    # Number of classes

        decoder = MLP(input_size=input_size, output_size=output_size)

        # Define a loss function and optimizer
        criterion = nn.CrossEntropyLoss()  # Instead of logistic regression, since logistic is for binary classification
        optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)

        decoder.fit(train_loader=decoder_dl, optimizer=optimizer, criterion=criterion, num_epochs=10)
        print("Done.")

        # Save decoder to checkpoint
        try:
            print("Saving decoder checkpoint...", end=' ')
            torch.save(decoder.state_dict(), f"{checkpoint_dir}/decoder.ckpt")
            print("Done.")
        except Exception as e:
            print(f"Failed to save decoder checkpoint: {e}")

    ## CREATE COMPOSITE MODEL
    pssm = PoissonSSM(encoder, decoder)
    return pssm, ds_test

def evaluate(pssm, ds_test):
   pssm.get_rts(ds_test[0][0], threshold=0.25, n_repeats=1000)

if __name__ == "__main__":
    print("Starting PSSM training pipeline...")
    pssm, ds_test = build()
    pssm.encoder.to('cpu')  # Ensure encoder is on CPU for evaluation
    pssm.decoder.to('cpu')  # Ensure decoder is on CPU for evaluation
    evaluate(pssm, ds_test)
    print("Done!")
