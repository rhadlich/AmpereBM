#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 14:46:00 2023

@author: rodrigohadlich
"""

"""
This code is meant for running on CPU only. To run this use the following 
command in terminal:

    OMP_NUM_THREADS=<num_threads> mpirun --map-by ppr:1:numa:pe=11 python3 mpi_new.py <save_every> <total_epochs> '/lustre/home/rristowhadli/hdf5_data' --num_layers=<num_layers> --num_nodes_exp=<layer_exp> --lr=<lr> --batch_size=32

--nproc_per_node tells torchrun how many threads to use in each node.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from LoaderCAD import get_dataloader, get_datashapes
import time

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import h5py
import comm

device = torch.device('cpu')
# print(f'The number of threads is {torch.get_num_threads()}')
# torch.set_num_threads(3)
# print(f'The number of threads is {torch.get_num_threads()}')


# %%
# ---------------- HYPERPARAMETERS ----------------#
input_size = 4  # number of features
scheduler_step = 20
scheduler_gamma = 0.1
criterion = nn.MSELoss(reduction='mean')
mae = nn.L1Loss()


# ---------------- Define model layers ----------------#
def model_layers(input_size, num_layers, layer_exp, P, out_size):
    in_features = input_size
    exp_layers = np.full(shape=num_layers, fill_value=layer_exp, dtype=int)
    layers = []
    P = 1 - P
    for i, exp in enumerate(exp_layers):
        n_units_FC = int((2 ** exp) / P)
        layers.append(nn.Linear(in_features, n_units_FC))
        layers.append(nn.Tanh())
        layers.append(nn.Dropout(p=(1 - P)))
        in_features = n_units_FC
    layers.append(nn.Linear(in_features, out_size))
    return layers


# %%
def ddp_setup(method):
    comm_local_group = comm.init(method)
    return comm_local_group


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            val_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            scheduler,
    ) -> None:
        self.local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        self.global_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        self.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs_run = 0
        self.loss = float('inf')
        self.model = model

        self.loss_logger = float('inf')
        self.val_loss = float('inf')

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        # source = source.float()
        # targets = targets.float()
        output = self.model(source)
        output = torch.squeeze(output)
        loss = criterion(output, targets)
        self.loss = loss
        self.loss_logger += loss
        loss.backward()
        self.optimizer.step()

    def _run_val(self, loader):
        self.model.eval()
        val_logger = 0
        mae_logger = 0
        with torch.no_grad():
            for val_idx, (source, targets, filename, idx) in enumerate(loader):
                source = source.to(device)
                targets = targets.to(device)
                output = self.model(source)
                output = torch.squeeze(output)
                targets = torch.squeeze(targets)
                loss = criterion(output, targets)
                mae_val = mae(output, targets)
                val_logger += loss
                mae_logger += mae_val
        self.val_loss = val_logger / (val_idx + 1)
        self.val_mae = mae_logger / (val_idx + 1)
        self.model.train()

    def _run_epoch(self, epoch):

        self.train_data.sampler.set_epoch(epoch)

        self.loss_logger = 0

        for batch_idx, (source, targets, filename, idx) in enumerate(self.train_data):
            source = source.to(device)
            targets = targets.to(device)
            self._run_batch(source, targets)

        if self.global_rank == 0:
            print(f"[Epoch {epoch} | Loss: {(self.loss_logger / (batch_idx + 1)):.4f} | Steps: {len(self.train_data)}")

        self.scheduler.step()

    def train(self, max_epochs):
        # run training loop

        time_epoch = 0
        for epoch in range(self.epochs_run, max_epochs):
            tic_epoch = time.time()
            self._run_epoch(epoch)
            toc_epoch = time.time()
            time_epoch += toc_epoch - tic_epoch

        # average time per epoch and reduce across processes
        time_epoch /= max_epochs
        time_epoch = torch.tensor(time_epoch / self.world_size)
        dist.all_reduce(time_epoch)
        if self.global_rank == 0:
            print(f'The average time per epoch is {time_epoch}s')
        self.time_epoch = time_epoch

        # evaluate model on validation set
        self._run_val(self.val_data)
        dist.all_reduce(self.val_loss)
        self.val_loss /= self.world_size
        if self.global_rank == 0:
            print(f'The MSE of the model on the validation set is {self.val_loss}.')


def main(total_epochs, root_dir, node_type, method, num_layers, layer_exp, learning_rate, batch_size, P, n_trials):
    # Initialize distributed process group
    comm_local_group = ddp_setup(method)

    # Get ranks and sizes
    rank = comm.get_rank()
    local_rank = comm.get_local_rank()
    size = comm.get_size()
    local_size = comm.get_local_size()

    # Get data loaders and data shape
    train_loader, train_size, validation_loader, validation_size = get_dataloader(root_dir, device, size, rank,
                                                                                  batch_size)
    data_shape, labels_shape = get_datashapes(root_dir)
    if rank == 0:
        print(f'Data Shape: {data_shape}')
        print(f'Labels Shape: {labels_shape}')
    out_size = 1  # Need to adjust this based on the format in hdf5 file

    # Create model and wrap it with DDP
    torch.manual_seed(123)
    layers = model_layers(input_size, num_layers, layer_exp, P, out_size)
    model = nn.Sequential(*layers)
    model = model.to(device)
    model = DDP(model,
                device_ids=None,
                output_device=None,
                )

    # save model weights
    filename_model = 'model_weights_' + node_type + '.pth'
    if rank == 0:
        torch.save(model.state_dict(), filename_model)

    # Set optimizer, scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate * size)  # Compensating for number of processes
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)            # Not considering number of processes
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    times_total = np.zeros([n_trials])
    times_epoch = np.zeros([n_trials])
    mse = np.zeros([n_trials])
    mae = np.zeros([n_trials])

    # save optimizer weights
    filename_optimizer = 'optimizer_weights_' + node_type + '.pth'
    if rank == 0:
        torch.save(optimizer.state_dict(), filename_optimizer)

    dist.barrier()

    # loop for running the same model multiple times and record times
    for i in range(n_trials):

        # Start training model
        model.load_state_dict(torch.load(filename_model))
        optimizer.load_state_dict(torch.load(filename_optimizer))
        model.train()
        trainer = Trainer(model, train_loader, validation_loader, optimizer, scheduler)
        start_time = time.gmtime(time.time())
        tic = time.time()
        trainer.train(total_epochs)
        toc = time.time()
        end_time = time.gmtime(time.time())
        t = toc - tic
        t = torch.tensor(t)
        loss = trainer.val_loss
        mae_loss = trainer.val_mae
        dist.all_reduce(t)
        dist.all_reduce(loss)
        dist.all_reduce(mae_loss)
        t /= size
        loss /= size
        mae_loss /= size

        # store time and losses into np array
        times_total[i] = t.numpy(force=True)
        times_epoch[i] = trainer.time_epoch.numpy(force=True)
        mse[i] = loss.numpy(force=True)
        mae[i] = mae_loss.numpy(force=True)

        # print things
        if rank == 0:
            print(
                f'Iteration: {i} | Model Training Time: {toc - tic} seconds | Number of Threads: {torch.get_num_threads()} | World Size: {size}')
            # torch.save(trainer.model.state_dict(), 'model_weights.pth')

    # save things in hdf5 file
    if rank == 0:
        current_dir = os.getcwd()
        file_dir = os.path.join(current_dir, 'ampere_bm.h5')
        with h5py.File(file_dir, 'a') as f:

            # check to see if file already existed/was populated
            if node_type not in f:
                # needs to create groups
                f.create_group(node_type)

            # at this point groups for model size exist
            grp1 = f[node_type]

            n_processes = str(size)

            # check to see if a group for the current number of processes already exists, if it does then delete it
            if n_processes in grp1:
                del grp1[n_processes]

            # create group for current number of processes
            grp2 = grp1.create_group(n_processes)

            # create datasets under current group
            grp2.create_dataset(name='total time',
                                shape=(n_trials,),
                                dtype='f',
                                data=times_total, )
            grp2.create_dataset(name='epoch time',
                                shape=(n_trials,),
                                dtype='f',
                                data=times_epoch, )
            grp2.create_dataset(name='start time',
                                shape=(n_trials,),
                                dtype='f',
                                data=start_time, )
            grp2.create_dataset(name='end time',
                                shape=(n_trials,),
                                dtype='f',
                                data=end_time, )
            grp2.create_dataset(name='mse',
                                shape=(n_trials,),
                                dtype='f',
                                data=mse, )
            grp2.create_dataset(name='mae',
                                shape=(n_trials,),
                                dtype='f',
                                data=mae, )

    # end distributed process group
    destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    # parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('root_dir', type=str,
                        help='Root directory where train, validation, and test folders are located')
    parser.add_argument('node_type', type=str, help='Whether it is SPR, Grace, Ampere, or AmpereOneX')
    parser.add_argument('method', type=str, help='Whether to use nccl or mpi backend')
    parser.add_argument('--num_layers', default=4, type=int, help='Number of hidden layers (default: 4)')
    parser.add_argument('--num_nodes_exp', default=8, type=int,
                        help='Exponential factor for 2**n nodes per layer (default: 8)')
    parser.add_argument('--lr', default=0.00064, type=float, help='Learning rate (default: 0.00064)')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--p', default=0.3, type=float, help='Dropout probability (default: 0.3)')
    parser.add_argument('--n_trials', default=30, type=int, help='Number of consecutive trials (default: 30)')
    args = parser.parse_args()

    main(args.total_epochs, args.root_dir, args.node_type, args.method, args.num_layers, args.num_nodes_exp, args.lr,
         args.batch_size, args.p, args.n_trials)

