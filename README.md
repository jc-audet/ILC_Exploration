
### ILC Exploration repo
This repo contains empirical exploration of the ILC methodology and attempts at improvement of the AND-Mask method through generalized combination of gradients


#### Instructions
To run the baseline (standard SGD), use `method='and_mask'` and `agreement_threshold=0.`

There are two examples:
### Synthetic dataset

#### AND-Mask

```
python3 -m and_mask.run_synthetic \
        --method=and_mask \
        --agreement_threshold=1. \
        --n_train_envs=16 \
        --n_agreement_envs=16 \
        --batch_size=256 \
        --n_dims=16 \
        --scale_grad_inverse_sparsity=1 \
        --use_cuda=1 \
        --n_hidden_units=256
```

#### GEN-Mask

```
python3 -m gen_mask.run_synthetic \
        --method=and_mask \
        --n_envs=16 \
        --batch_size=256 \
        --n_dims=16 \
        --use_cuda=1 \
        --n_hidden_units=256
```

### CIFAR-10

#### AND-Mask

```
python -m and_mask.run_cifar \
        --random_labels_fraction 1.0 \
        --agreement_threshold 0.2 \
        --method and_mask \
        --epochs 80 \
        --weight_decay 1e-06 \
        --scale_grad_inverse_sparsity 1 \
        --init_lr 0.0005 \
        --weight_decay_order before \
        --output_dir /tmp/
        --agreement_method=maj_vote \
```
#### GEN-Mask

```
python -m and_mask.run_cifar \
        --random_labels_fraction 1.0 \
        --agreement_threshold 0.2 \
        --method and_mask \
        --epochs 80 \
        --weight_decay 1e-06 \
        --scale_grad_inverse_sparsity 1 \
        --init_lr 0.0005 \
        --weight_decay_order before \
        --output_dir /tmp/
        --agreement_method=maj_vote \
```

### Landscape and gradient vizualization

See the vizualization folder.
