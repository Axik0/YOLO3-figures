"""Contains main training algorithm and related functions
this script is intended to be imported and run from jupyter notebook
"""
import os
from contextlib import nullcontext
from generation import PATH, AMOUNT

import torch
from tqdm.notebook import tqdm, tnrange  # otherwise it keeps adding new lines at each iteration
from torch.utils.tensorboard import SummaryWriter
from modules import LOSS_PARTS


CFP = os.path.join(PATH, 'checkpoints')
CH_NAME = 'last_state.pt'
LG_NAME = 'last_state.pt'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PALETTE = ("#eca0ff", "#aab2ff", "#84ffc9")  # custom tqdm colour palette


def splits(p, cap_length=AMOUNT):
    """takes subset of limit_length, generates 2 slices splitting it on two non-intersecting consequent parts,
        proportion p relates to the proportion of first part to limit_length, the rest is 2nd part"""
    assert 0 < p <= 1, f'Provided {p} proportion is incorrect'
    assert 0 < cap_length <= AMOUNT, f'Provided {cap_length} is beyond limits'
    return slice(None, int(cap_length * p)), slice(int(cap_length * p), cap_length)


def save_ch(model, optimizer, curr_loss, curr_epoch, folder_path=CFP, name=CH_NAME):
    """saves internal state of model and optimizer, !huge! requires 0.5Gb space!"""
    checkpoint = {'epoch': curr_epoch,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'loss': curr_loss, }
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        pass
    finally:
        torch.save(obj=checkpoint, f=os.path.join(folder_path, name))
        return True


def logger(writer, curr_epoch, train_loss_state, test_loss_state=None):
    """dumps train (and test as option) loss states to file that could be viewed via TensorBoard,
        writer should be an instance of tensorboard's SummaryWriter
        """
    curr_train_log = {LOSS_PARTS[i]: train_loss_state[i] for i in range(len(train_loss_state))}
    writer.add_scalars(main_tag='train_loss', tag_scalar_dict=curr_train_log, global_step=curr_epoch)
    if test_loss_state:
        curr_test_log = {LOSS_PARTS[i]: test_loss_state[i] for i in range(len(test_loss_state))}
        writer.add_scalars(main_tag='test_loss', tag_scalar_dict=curr_test_log, global_step=curr_epoch)
    writer.close()
    return True


def load_ch(model, optimizer, path=os.path.join(CFP, CH_NAME)):
    """loads previous states into model and optimizer from disk"""
    try:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f'Checkpoint at epoch {epoch} with loss {loss:.2f} has been loaded')
        return int(epoch)
    except FileNotFoundError:
        print(f'no checkpoint there {path}')


def run(model, dataloader, loss_fn, amp_scaler=None, optimizer=None, device=DEVICE, agg=True):
    """universal (forward + optional backward) pass, requires tqdm-wrapped dataloader
    loss component mean aggregation (over batch dimension) is set up as a default output,
    """
    states = []
    model.train() if optimizer else model.eval()
    with nullcontext() if optimizer else torch.inference_mode():
        for img, tar in dataloader:
            x, y = img.to(device), (tar[0].to(device), tar[1].to(device), tar[2].to(device))
            # forward pass within mixed precision context (casts to lower precision dtype if possible)
            with torch.autocast(device) if amp_scaler else nullcontext():
                p = model(x)
                # tricky: loss_fn forward call must happen BEFORE get_state() and that's how it is by default in python
                loss_data = [(loss_fn(pred_s=p[s], tar_s=y[s], scale=s), loss_fn.get_state()) for s in range(3)]
                loss, state = tuple(map(lambda tl: torch.sum(torch.stack(tl, dim=0), dim=0), zip(*loss_data)))
            states.append(state)  # tensor of 4 elements (already averaged over current batch)
            # backward pass
            if optimizer:
                optimizer.zero_grad()
                if amp_scaler and device != 'cpu':  # GradScaler doesn't support CPU
                    # scales loss before backprop to help mixed precision
                    amp_scaler.scale(loss).backward()
                    # unscales gradient steps within optimizer
                    amp_scaler.step(optimizer)
                    # tries larger scale (if inf => skip, revert)
                    amp_scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            # cumulative averaging approach
            ba_loss_dist = torch.mean(torch.stack(states, dim=0), dim=0)  # loss distribution (mean over batch dim)
            dataloader.set_postfix_str(f'Current loss {torch.sum(ba_loss_dist).item():.2e}', refresh=True)
        return ba_loss_dist if agg else states


def train(model, dataloader_train, loss_fn, optimizer, n_epochs,
          amp_scaler, device=DEVICE, dataloader_test=None, ep=2, load=False, palette=PALETTE):
    """Model training procedure w/ possible evaluation (if test data is provided) and autosave/autoload
        ep -- model evaluation period, any integer >= 0,
        amp_scaler should be an instance of torch.cuda.amp.GradScaler for automated mixed precision training on GPU"""
    loaded_epoch = load_ch(model, optimizer) if load else 0     # inplace last state loader
    assert loaded_epoch < n_epochs, f'Set number of epochs larger than loaded, {loaded_epoch}'

    tb_writer = SummaryWriter(CFP)

    model = model.to(device=device)
    loss_fn = loss_fn.to(device=device)    # as it's stateful, has built-in hyperparameters (weighing parts of loss)

    epochs_ = tnrange(*(loaded_epoch, n_epochs), desc='Epoch: ', colour=palette[2], position=0, leave=True)
    for e in epochs_:
        # main forward + backward pass
        dataloader_train_ = tqdm(dataloader_train, desc='  Batch: ', colour=palette[1], position=1, leave=False)
        avg_train_loss_dist = run(model, dataloader_train_, loss_fn, amp_scaler=amp_scaler, optimizer=optimizer, device=device)
        avg_train_loss = torch.sum(avg_train_loss_dist).item()
        # optional evaluation (forward pass only)
        if (e + 1) % ep == 0 and dataloader_test is not None:
            dataloader_test_ = tqdm(dataloader_test, desc='Testing: ', colour=palette[0], position=2, leave=False)
            avg_test_loss_dist = run(model, dataloader_test_, loss_fn, amp_scaler=amp_scaler, device=device)
            avg_test_loss = torch.sum(avg_test_loss_dist).item()
            ceu_str = (f'test loss {avg_test_loss:.2f}',
                       f', distributed as {tuple(map(lambda x:round(x, 2), avg_test_loss_dist.tolist()))}')
            # TensorBoard logging to file
            logger(tb_writer, e, avg_train_loss_dist.tolist(), avg_test_loss_dist.tolist())
        else:
            ceu_str = (f'train loss {avg_train_loss:.2f}',
                       f', distributed as {tuple(map(lambda x:round(x, 2), avg_train_loss_dist.tolist()))}')
        epochs_.set_postfix_str(ceu_str[0], refresh=True)
        # autosave checkpoint after each epoch
        saved = save_ch(model, optimizer, curr_loss=avg_train_loss, curr_epoch=e+1)
        if saved:
            epochs_.write(f'checkpoint after {e + 1} epoch saved, ' + ceu_str[0] + ceu_str[1])


if __name__ == '__main__':
    from model import YOLO3
    from modules import FiguresDataset, YOLOLoss
    from torch.utils.data import DataLoader as DL
    LR = 1E-3

    # loading data
    ds_train = FiguresDataset(part=(None, 1000))
    ds_test = FiguresDataset(part=(1000, 1200))
    train_l = DL(dataset=ds_train, batch_size=4, shuffle=True)
    test_l = DL(dataset=ds_test, batch_size=4, shuffle=False)
    # instantiating
    y_model = YOLO3()
    y_loss = YOLOLoss()
    optim = torch.optim.Adam(y_model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler()
