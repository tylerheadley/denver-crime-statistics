# System Imports
import argparse
import os
import random
import sys
import time

# Library Imports
import numpy as np
import torch

#-------
# Args
#-------

def get_args():
    parser = argparse.ArgumentParser(description="A simple argument parser example")
    args = parser.parse_args()
    return args


#-------
# Model
#-------

class Model(Module):
    def __init__(self,)
        super(Model, self).__init__()
    def forward(self):
	    return None


#---------
# Helpers
#---------

def setup(args):
    # Set device
    args['device'] = args['device'] if torch.cuda.is_available() else 'cpu'
    # Set Backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Set Seeds
    seed = args['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    pass


def load(cfg):
    #train_dl, val_dl, test_dl = data_loaders()
    model = Model()
    return model, train_dl, val_dl, test_dl


#---------------------
# Train/Validate/Test
#---------------------

def train(criterion, data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(data.x)
    loss = criterion(output, data.label)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def validate(criterion, data, model):
    model.eval()
    output = model(data.x)
    loss = criterion(output, data.label)
    return loss.item()

@torch.no_grad()
def test(criterion, data, model):
    model.eval()
    output = model(data.x)
    loss = criterion(output, data.label)
    return loss.item()


#-----------------------
# Main/Fold/Train
#-----------------------

def run_training(args, model, train_dl, val_dl):

    # Transfer to Device
    model = model.to(cfg.setup['device'])
    for data in train_dl:
	data = data.to(cfg.setup['device'])
    for data in val_dl:
	data = data.to(cfg.setup['device'])

    # Set Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args['epochs'])
    criterion = F.nll_loss()

    # Train per Epoch
    best = 1e8
    for epoch in range(args['epochs']):

        model.train()
        train_loss, count = 0, 0
        start = time.time()

        # Train per Batch
        for i,data in enumerate(train_dl):
            batch_loss = train(criterion, data, model, optimizer, meann, mad)

            batch_size = data.y.shape[0]
            train_loss += batch_loss * batch_size
            count += batch_size

            if i%10 == 0:
                print(f'Train({epoch}) | batch({i:03d}) | loss({batch_loss:.4f})')

        end = time.time()
        train_loss = train_loss/count
        scheduler.step()

        # Validate per Batch
        model.eval()
        val_loss, count = 0, 0
        for i,data in enumerate(val_dl):
            batch_loss = validate(cfg, data, model, meann, mad)

            batch_size = data.y.shape[0]
            val_loss += batch_loss * batch_size
            count += batch_size

            if i%10 == 0:
                print(f'Valid({epoch}) | batch({i:03d}) | loss({batch_loss:.4f})')

        val_loss = val_loss/count
        perf_metric = val_loss #your performance metric here
        lr = optimizer.param_groups[0]['lr']

        # Save Best Performing Model
        if perf_metric < best:
            best = perf_metric
            bad_itr = 0
            torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': lr,
                'loss': val_loss,
                },
                cfg.load['checkpoint_path']
            )
        else:
            bad_itr += 1

        print(f'Epoch({epoch}) '
            f'| train({train_loss:.4f}) '
            f'| val({val_loss:.4f}) '
            f'| lr({lr:.2e}) '
            f'| best({best:.4f}) '
            f'| time({end-start:.4f})'
            f'\n')

        if bad_itr>args['patience']:
            break

    return best


#----------------------------------------------------------------------------------------------------------------------------------------------------

def run(args):
    # Load
    args = get_args()
    setup(args)
    model, train_dl, val_dl, test_dl = load(args)
    print(model)

    # Train
    run_training(cfg, model, train_dl, val_dl)

    # Load Best Model
    checkpoint = torch.load(cfg.load['checkpoint_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(cfg.setup['device'])

    for data in test_dl:
        data.to(cfg.setup['device'])

    # Test Best Model
    test_loss, count = 0, 0
    for data in test_dl:
        batch_loss = test(cfg, data, model, meann, mad)

        batch_size = data.y.shape[0]
        test_loss += batch_loss * batch_size
        count += batch_size
    test_loss = test_loss/count

    print(f'\ntest({test_loss})')
    return 1

#----------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    run()
