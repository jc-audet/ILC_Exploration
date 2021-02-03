import sys
import argparse

import numpy as np
from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.models import vgg16

from synbols.data_io import load_h5, load_attributes_h5
from and_mask.utils.ignite_cifar10_utils import get_train_test_loaders, get_model

from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset

from and_mask.and_mask_utils import get_grads
from and_mask.datasets.common import permutation_groups
import and_mask.datasets.synthetic.dataloader as synthetic_dataloader
from and_mask.models.synthetic import get_synthetic_model
from and_mask.utils.utils import add_l1_grads, validate_target_outupt_shapes, count_correct
import matplotlib.pyplot as plt

print(torch.version.cuda)
print("CUDA:", torch.cuda.is_available(), "Num. gpus:", torch.cuda.device_count())

# Function to get the char of an image
def get_char(y):
    return y['char']

# We define our Pytorch Dataset
class SynbolsDataset(Dataset):
    def __init__(self, x, y, transform):
        self.x, self.y, self.transform = x,y,transform
        self.y = [get_char(yi) for yi in self.y]
        self._all_chars = sorted(list(set(self.y)))
        
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        x,y = self.x[idx], self.y[idx]
        return self.transform(x), self._all_chars.index(y)

# Define transforms
train_tf = T.Compose([T.ToPILImage(), T.ToTensor()])
test_tf = T.Compose([T.ToPILImage(), T.ToTensor()])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--examples_per_env', type=int, default=1024)
    parser.add_argument('--n_test', type=int, default=2000)
    parser.add_argument('--n_train_envs', type=int, default=16)
    parser.add_argument('--n_agreement_envs', type=int, default=16)
    parser.add_argument('--n_revolutions', type=int, default=3)
    parser.add_argument('--n_dims', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--method', type=str, choices=['and_mask', 'geom_mean', 'harmonic_mean', 'median'], required=True)
    parser.add_argument('--scale_grad_inverse_sparsity', type=int, choices=[0, 1], required=True)
    parser.add_argument('--agreement_threshold', type=float, required=True)
    parser.add_argument('--agreement_method', type=str, choices=['gini_ind', 'maj_vote', 'entropy'], required=True)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--batch_norm', type=int, default=0, choices=[0, 1])
    parser.add_argument('--dropout_p', type=float, default=0.0)
    parser.add_argument('--n_hidden_units', type=int, default=256)
    parser.add_argument('--n_hidden_layers', type=int, default=3)
    parser.add_argument('--l1_coef', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use_cuda', type=int, default=1, choices=[0, 1])
    parser.add_argument('--confidence', action='store_true')
    return parser.parse_args()


def train(model, device, train_loaders, optimizer, epoch, writer,
          scale_grad_inverse_sparsity,
          n_agreement_envs,
          loss_fn,
          l1_coef,
          method,
          agreement_threshold,
          agreement_method,
          confidence,
          scheduler,
          log_suffix=''):
    """n_agreement_envs is the number of envs used to compute agreements"""
    assert len(train_loaders) % n_agreement_envs == 0  # Divisibility makes it more convenient
    model.train()

    losses = []
    correct = 0
    example_count = 0
    batch_idx = 0

    train_iterators = [iter(loader) for loader in train_loaders]
    it_groups = permutation_groups(train_iterators, n_agreement_envs)

    while 1:
        train_iterator_selection = next(it_groups)
        try:
            datas = [next(iterator) for iterator in train_iterator_selection]
        except StopIteration:
            break

        assert len(datas) == n_agreement_envs

        batch_size = datas[0][0].shape[0]
        assert all(d[0].shape[0] == batch_size for d in datas)

        inputs = [d[0].to(device) for d in datas]
        target = [d[1].to(device) for d in datas]
        inputs = torch.cat(inputs, dim=0)
        target = torch.cat(target, dim=0)

        optimizer.zero_grad()

        output = model(inputs)
        output = output.squeeze(1)
        validate_target_outupt_shapes(output, target)

        mean_loss, masks = get_grads(
            agreement_threshold,    
            agreement_method,
            confidence,
            batch_size,
            loss_fn, n_agreement_envs,
            params=optimizer.param_groups[0]['params'],
            output=output,
            target=target,
            method=method,
            scale_grad_inverse_sparsity=scale_grad_inverse_sparsity,
        )

        if l1_coef > 0.0:
            add_l1_grads(l1_coef, optimizer.param_groups)

        optimizer.step()

        losses.append(mean_loss.item())
        correct += count_correct(output, target)
        example_count += output.shape[0]
        batch_idx += 1

    # Logging
    train_loss = np.mean(losses)
    train_acc = correct / (example_count + 1e-10)
    writer.add_scalar(f'weight/norm', train_loss, epoch)
    writer.add_scalar(f'mean_loss/train{log_suffix}', train_loss, epoch)
    writer.add_scalar(f'acc/train{log_suffix}', train_acc, epoch)
    logger.info(f'Train Epoch: {epoch}\t Acc: {train_acc:.4} \tLoss: {train_loss:.6f}')


def run_test(model, device, test_loader, writer, epoch, loss_fn, scheduler, log_suffix=''):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            output = output.squeeze(1)

            validate_target_outupt_shapes(output, target)

            test_loss += loss_fn(output, target).item()  # sum up batch loss
            correct += count_correct(output, target)
            total += data.shape[0]

    scheduler.step(test_loss)

    test_acc = correct / total
    logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, total,
        100. * test_acc))

    writer.add_scalar(f'loss/test{log_suffix}', test_loss, epoch)
    writer.add_scalar(f'acc/test{log_suffix}', test_acc, epoch)


def main(args):
    device = torch.device("cuda" if args.use_cuda else "cpu")

    kwargs = {'num_workers': 0, 'pin_memory': True} if args.use_cuda else {}

    train_envs = list(range(args.n_train_envs))
    train_loaders = []

    # Create our Synbols datasets
    x_val, mask_val, y_val, split_val = load_h5('./and_mask/datasets/synbols/validation.h5py')
    # x_val, mask_val, y_val, split_val = load_h5('./and_mask/datasets/synbols/train_0.h5py')
    val_ds = SynbolsDataset(x_val, y_val, transform=test_tf)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

    x_test, mask_test, y_test, split_test = load_h5('./and_mask/datasets/synbols/test.h5py')
    # x_test, mask_test, y_test, split_test = load_h5('./and_mask/datasets/synbols/train_0.h5py')
    test_ds = SynbolsDataset(x_test, y_test, transform=test_tf)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

    for env in train_envs:

        print("Loading train environment #" + str(env))
        x_train, mask_train, y_train, split_train = load_h5('./and_mask/datasets/synbols/train_' + str(env) + '.h5py')
        home_dl = SynbolsDataset(x_train, y_train, transform=train_tf)
        dl = torch.utils.data.DataLoader(home_dl, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        train_loaders.append(dl)

    loss_fn = nn.CrossEntropyLoss()

    print('./runs/' + args.method + '_' + str(args.agreement_threshold))
    summary_writer = SummaryWriter('./runs/' + args.method + '_' + str(args.agreement_threshold))

    model = get_model(num_classes=96)
    model = model.to(device)

    weight_decay = args.weight_decay

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                            weight_decay=weight_decay,
                            nesterov=True, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)


    logger.info('===================')
    for key, val in vars(args).items():
        logger.info(f'  {key}: {val}')

    for epoch in range(1, args.epochs + 1):
        run_test(model, device, test_loader, summary_writer, epoch,
                 loss_fn=loss_fn, scheduler=scheduler,
                 log_suffix='_probe')

        train(model,
              device,
              train_loaders,
              optimizer,
              epoch,
              summary_writer,
              scale_grad_inverse_sparsity=args.scale_grad_inverse_sparsity,
              n_agreement_envs=args.n_agreement_envs,
              loss_fn=loss_fn,
              l1_coef=args.l1_coef,
              method=args.method,
              agreement_threshold=args.agreement_threshold,
              agreement_method=args.agreement_method,
              confidence=args.confidence,
              scheduler=scheduler,
              log_suffix='_probe',
              )

        if summary_writer is not None:
            summary_writer.flush()


if __name__ == '__main__':
    logger.remove(0)
    logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>")
    torch.manual_seed(0)
    np.random.seed(0)
    args = parse_args()
    main(args)
