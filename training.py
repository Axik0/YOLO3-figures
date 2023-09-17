"""training and related scripts"""
import tqdm
import torch
from torch.utils.data import DataLoader as DL
from contextlib import nullcontext

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import YOLO3
from process import FiguresDataset, YOLOLoss

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_TRANSFORMS = [A.Normalize((0, 0, 0), (0.5, 0.5, 0.5)), A.Resize(416, 416), ToTensorV2()]
# no augmentations for now: Normalized images, resized 256 --> 416, cast to torch.float tensors...(NB! order matters)


def run(model, dataloader, loss_fn, optimizer=None, device=DEVICE, agg=True):
    """single universal (forward + optional backward) pass, optional mean aggregation, MA over examples"""
    losses = []
    model.train() if optimizer else model.eval()
    with nullcontext() if not optimizer else torch.inference_mode():
        for img, tar in dataloader:
            x, y = img.to(device), (tar[0].to(device), tar[1].to(device), tar[3].to(device))
            # forward pass
            p = model(x)
            loss_sc = [loss_fn(pred_s=p[s], tar_s=y[s], scale=s) for s in range(3)]
            loss = torch.sum(torch.stack(loss_sc, dim=0), dim=0)
            losses.append(loss.item())
            # backward pass
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return torch.mean(torch.stack(losses, dim=0)).item() if agg else losses


def train(model, dataloader_train, dataloader_test, loss_fn, optimizer, n_epochs, device=DEVICE):
    epochs_ = tqdm.trange(n_epochs, desc='Epoch: ', position=0)
    dataloader_ = tqdm.tqdm(dataloader_train, colour='green', position=1)
    ss = 100  # description update period
    for e in epochs_:
        avg_train_loss = run(model, dataloader_, loss_fn, optimizer=optimizer, device=device, agg=True)
        if e % ss == 0:
            avg_test_loss = run(model, dataloader_test, loss_fn, device=device, agg=True)
            dataloader_.set_postfix_str(f'Train loss {avg_train_loss:.5f}', refresh=True)
            epochs_.set_description(f'Test loss {avg_test_loss:.5f}', refresh=True)


if __name__ == '__main__':
    dtr = A.Compose(DEFAULT_TRANSFORMS,
                    bbox_params=A.BboxParams(format='yolo', label_fields=['cidx'], min_visibility=0.5))
    ds = FiguresDataset(transforms=dtr)
    yolo = YOLO3().to(device=DEVICE)
    loss_f = YOLOLoss()
    train_l = DL(dataset=ds, batch_size=1, shuffle=True)
    test_l = DL(dataset=ds, batch_size=1, shuffle=False)
    optimizer = torch.optim.Adam(yolo.parameters(), lr=1e-5)

    train(yolo, train_l, test_l, loss_f, optimizer, device=DEVICE, n_epochs=300)
