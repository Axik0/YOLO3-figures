"""training and related scripts"""
from tqdm.notebook import tqdm, tnrange
import torch
from torch.utils.data import DataLoader as DL
from contextlib import nullcontext

import os
from generation import PATH
CFP = os.path.join(PATH, 'checkpoints')
CH_NAME = 'last_state.pt'


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def save_ch(model, optimizer, curr_loss, curr_epoch, folder_path=CFP, name=CH_NAME):
    """saves state of model and optimizer"""
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


def load_ch(model, optimizer, path=os.path.join(CFP, CH_NAME)):
    """loads previous state to model, optimizer"""
    try:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except FileNotFoundError:
        print(f'no checkpoint there {path}')


def run(model, dataloader, loss_fn, scaler, optimizer=None, device=DEVICE, agg=True):
    """single universal (forward + optional backward) pass, mean aggregation over dataset by default as an output"""
    losses = []
    model.train() if optimizer else model.eval()
    with nullcontext() if optimizer else torch.inference_mode():
        for img, tar in dataloader:
            x, y = img.to(device), (tar[0].to(device), tar[1].to(device), tar[2].to(device))
            # forward pass within mixed precision context (casts to lower precision if possible)
            with torch.autocast(device) if scaler else nullcontext():
                p = model(x)
                loss_sc = [loss_fn(pred_s=p[s], tar_s=y[s], scale=s) for s in range(3)]
                loss = torch.sum(torch.stack(loss_sc, dim=0), dim=0)
            losses.append(loss.item())
            # backward pass
            if optimizer:
                optimizer.zero_grad()
                # scales loss before backprop
                if scaler and device != 'cpu':
                    scaler.scale(loss).backward()
                    # unscales gradient steps within optimizer
                    scaler.step(optimizer)
                    # tries larger scale (if inf => skip, revert)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            dataloader.set_postfix_str(f'Train loss {losses[-1]:.2e}', refresh=True)
        return torch.mean(torch.stack(losses, dim=0)).item() if agg else losses


def train(model, dataloader_train, dataloader_test, loss_fn, optimizer, n_epochs, scaler=None, device=DEVICE):
    """model training w/ evaluation on test dataset"""
    model = model.to(device=device)
    epochs_ = tnrange(n_epochs, desc='Epoch: ', position=0, leave=True)
    dataloader_train_ = tqdm(dataloader_train, desc='Batch: ', colour='green', position=1, leave=True)
    ss = 2  # description update period
    for e in epochs_:
        avg_train_loss = run(model, dataloader_train_, loss_fn, scaler=scaler, optimizer=optimizer, device=device)
        saved = save_ch(model, optimizer, curr_loss=avg_train_loss, curr_epoch=e)
        if saved:
            epochs_.write('checkpoint at {e} epoch saved')
        if e % ss == 0:
            avg_test_loss = run(model, dataloader_test, loss_fn, scaler=scaler, device=device)
            epochs_.set_description(f'Test loss {avg_test_loss:.2e}', refresh=True)


if __name__ == '__main__':
    from model import YOLO3
    from modules import FiguresDataset, YOLOLoss

    LR = 1E-5
    NUM_EPOCHS = 300

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
    # actual processing
    # train(y_model, train_l, test_l, y_loss, optim, device=DEVICE, n_epochs=NUM_EPOCHS)
