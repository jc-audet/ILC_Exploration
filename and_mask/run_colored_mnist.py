import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from loguru import logger
import and_mask.and_mask_utils as and_mask_utils

from datasets.colored_mnist.colored_mnist_utils import get_train_test_loaders
from models.mnist import get_mnist_model
from optimizers.adam_flexible_weight_decay import AdamFlexibleWeightDecay

from and_mask.utils.utils import add_l1_grads, add_l2_grads, add_l1, add_l2, validate_target_outupt_shapes, count_correct
from torch.utils.tensorboard import SummaryWriter
import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--data_path', type=str, default="/tmp/cifar_dataset")
    parser.add_argument('--agreement_threshold', type=float, required=True)
    parser.add_argument('--weight_decay', type=float, required=True)
    parser.add_argument('--method', type=str, choices=['and_mask', 'geom_mean'], required=True)
    parser.add_argument('--scale_grad_inverse_sparsity', type=int, required=True)
    parser.add_argument('--init_lr', type=float, required=True)
    parser.add_argument('--random_labels_fraction', type=float, required=True)
    parser.add_argument('--weight_decay_order', type=str,
                        choices=['before', 'after'], default='before')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--use_cuda', type=int, default=0, choices=[0, 1])
    parser.add_argument('--batch_norm', type=int, default=0, choices=[0, 1])
    parser.add_argument('--dropout_p', type=float, default=0.0)
    parser.add_argument('--n_dims', type=int, default=3*28*28)
    parser.add_argument('--n_hidden_units', type=int, default=28*28)
    parser.add_argument('--n_hidden_layers', type=int, default=0)
    parser.add_argument('--n_outputs', type=int, default=10)
    parser.add_argument('--l1_coef', type=float, default=0.0)
    parser.add_argument('--l2_coef', type=float, default=0.0)
    return parser.parse_args()

def run_test(model, args, device, test_loader, writer, epoch, loss_fn, log_suffix=''):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.reshape(-1, args.n_dims)
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            output = output.squeeze(1)

            validate_target_outupt_shapes(output, target)

            test_loss += loss_fn(output, target).item()  # sum up batch loss
            correct += count_correct(output, target)
            total += data.shape[0]

    if total != 0:
        test_acc = correct / total
    else:
        test_acc = 0
        
    logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, total,
        100. * test_acc))

    writer.add_scalar(f'loss/test{log_suffix}', test_loss, epoch)
    writer.add_scalar(f'acc/test{log_suffix}', test_acc, epoch)


def train(model, args, device, train_loader, optimizer, epoch, writer,
          scale_grad_inverse_sparsity,
          loss_fn,
          method,
          agreement_threshold,
          scheduler,
          log_suffix=''):
    model.train()
    losses = []
    correct = 0
    example_count = 0
    batch_idx = 0

    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, args.n_dims)
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        y_pred = model(images)
        if agreement_threshold > 0.0:
            # The "batch_size" in this function refers to the batch size per env
            # Since we treat every example as one env, we should set the parameter
            # n_agreement_envs equal to batch size
            mean_loss, masks = and_mask_utils.get_grads(
                agreement_threshold=agreement_threshold,
                batch_size=1,
                loss_fn=loss_fn,
                n_agreement_envs=args.batch_size,
                params=optimizer.param_groups[0]['params'],
                output=y_pred,
                target=labels,
                method=args.method,
                scale_grad_inverse_sparsity=scale_grad_inverse_sparsity,
            )
        else:
            mean_loss = loss_fn(y_pred, labels)
            mean_loss.backward()

        mean_total_loss = 0

        if args.l1_coef > 0.0:
            add_l1_grads(args.l1_coef, optimizer.param_groups)
            mean_total_loss += add_l1(args.l1_coef, optimizer.param_groups)
        if args.l2_coef > 0.0:
            add_l2_grads(args.l2_coef, optimizer.param_groups)
            mean_total_loss += add_l2(args.l2_coef, optimizer.param_groups)

        mean_total_loss += mean_loss.item()

        optimizer.step()
        
        losses.append(mean_total_loss)
        correct += count_correct(y_pred, labels)
        example_count += y_pred.shape[0]
        batch_idx += 1

    scheduler.step()

    # Logging
    train_loss = np.mean(losses)
    train_acc = correct / (example_count + 1e-10)
    writer.add_scalar(f'weight/norm', train_loss, epoch)
    writer.add_scalar(f'mean_loss/train{log_suffix}', train_loss, epoch)
    writer.add_scalar(f'acc/train{log_suffix}', train_acc, epoch)
    logger.info(f'Train Epoch: {epoch}\t Acc: {train_acc:.4} \tLoss: {train_loss:.6f}')

def main(args):
    device = torch.device("cuda" if args.use_cuda else "cpu")
    config = vars(args)

    (train_loader,
     test_loader,
     mislabeled_train_loader) = get_train_test_loaders(
        path=config['data_path'],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_seed=config['seed'],
        random_labels_fraction=config['random_labels_fraction'],
    )

    time = datetime.datetime.now()
    summary_writer = SummaryWriter(f'./tmp/seed_{args.seed}/{time}')

    model_original = get_mnist_model(args, device)
    model_random = get_mnist_model(args, device)

    optimizer_original = AdamFlexibleWeightDecay(model_original.parameters(),
                                        lr=config['init_lr'],
                                        weight_decay_order=config['weight_decay_order'],
                                        weight_decay=config['weight_decay'])
    
    optimizer_random = AdamFlexibleWeightDecay(model_random.parameters(),
                                    lr=config['init_lr'],
                                    weight_decay_order=config['weight_decay_order'],
                                    weight_decay=config['weight_decay'])

    criterion = nn.CrossEntropyLoss().to(device)

    le = len(train_loader)
    scheduler_original = MultiStepLR(optimizer_original,
                               milestones=[le * config['epochs'] * 3 // 4],
                               gamma=0.1)

    scheduler_random = MultiStepLR(optimizer_random,
                            milestones=[len(mislabeled_train_loader) * config['epochs'] * 3 // 4],
                            gamma=0.1)

    logger.info('===================')
    for key, val in vars(args).items():
        logger.info(f'  {key}: {val}')


    for epoch in range(1, args.epochs + 1):
        if config['random_labels_fraction'] > 0.0:
            run_test(model_random, args, device, test_loader, summary_writer, epoch,
                loss_fn=criterion,
                log_suffix='_probe_mis')
            train(model_random,
                args,
                device,
                mislabeled_train_loader,
                optimizer_random,
                epoch,
                summary_writer,
                scale_grad_inverse_sparsity=args.scale_grad_inverse_sparsity,
                loss_fn=criterion,
                method=args.method,
                agreement_threshold=args.agreement_threshold,
                scheduler=scheduler_random,
                log_suffix='_probe_mis',
                )

        else:
            run_test(model_original, args, device, test_loader, summary_writer, epoch,
                    loss_fn=criterion,
                    log_suffix='_probe')
            train(model_original,
                args,
                device,
                train_loader,
                optimizer_original,
                epoch,
                summary_writer,
                scale_grad_inverse_sparsity=args.scale_grad_inverse_sparsity,
                loss_fn=criterion,
                method=args.method,
                agreement_threshold=args.agreement_threshold,
                scheduler=scheduler_original,
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