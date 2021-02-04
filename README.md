
# ILC Exploration repo
This repo contains empirical exploration of the ILC methodology and attempts at improvement of the AND-Mask method through generalized combination of gradients.

This is a fork from the "Learning Explanations hat are Hard to Vary" paper repository. Credit to Amin M (@amimem) for the colored MNIST experiment on ILC

### Generalized-Mask (GEN-Mask)

Easy testing of different optimization of gradient combination can be explored in gen_mask/gen_mask_utils where the get_grads function is the function called at each training step

### TODO

 - [] Implement Benchmark from "Empirical or Invariant Risk Minimization? A Sample Complexity Perspective"

### Instructions
To run the baseline (standard SGD), use `method='and_mask'` and `agreement_threshold=0.`

## Running the experiments

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

### Colored MNIST

#### AND-Mask

```
python3 -m and_mask.run_colored_mnist \
        --method=and_mask \
        --output_dir=./data/ \
        --agreement_threshold=1. \
        --batch_size=256 \
        --random_labels_fraction=0.25 \
        --scale_grad_inverse_sparsity=1 \
        --use_cuda=1 \
        --n_hidden_units=256 \
        --init_lr=0.01 \
        --weight_decay=1e-3
```

#### GEN-Mask

```
python3 -m gen_mask.run_colored_mnist \
        --method=and_mask \
        --output_dir=./data/ \
        --batch_size=256 \
        --random_labels_fraction=0.25 \
        --use_cuda=1 \
        --n_hidden_units=256 \
        --init_lr=0.01 \
        --weight_decay=1e-3
```


### CIFAR-10

#### AND-Mask

```
python3 -m and_mask.run_cifar \
        --random_labels_fraction 1.0 \
        --agreement_threshold 0.2 \
        --method and_mask \
        --epochs 80 \
        --weight_decay 1e-06 \
        --scale_grad_inverse_sparsity 1 \
        --init_lr 0.0005 \
        --weight_decay_order before \
        --output_dir /tmp/
```
#### GEN-Mask

```
python -m gen_mask.run_cifar \
        --random_labels_fraction 1.0 \
        --method and_mask \
        --epochs 80 \
        --weight_decay 1e-06 \
        --init_lr 0.0005 \
        --weight_decay_order before \
        --output_dir /tmp/
```

### Landscape and gradient vizualization

See the vizualization folder.
