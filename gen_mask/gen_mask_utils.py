import torch
import numpy as np
from scipy.optimize import minimize

def get_grads(batch_size, n_envs, 
              loss_fn, params,
              output, target
              ):
    """
    General function that extracts environment-wise gradient and combine them for a final global gradient

    Args:
        batch_size: The original batch size per environment. Needed to perform reshaping, so that grads can be computed
            independently per each environment.
        n_envs: the number of environments that were stacked in the inputs. Needed to perform reshaping.
        loss_fn: the loss function
        params: the model parameters
        output: the output of the model, where inputs were *all examples from all envs stacked in a big batch*. This is
            done to at least compute the forward pass somewhat efficiently.

    Returns:
    """

    param_gradients = [[] for _ in params]
    outputs = output.view(n_envs, batch_size, -1)
    targets = target.view(n_envs, batch_size, -1)

    outputs = outputs.squeeze(-1)
    targets = targets.squeeze(-1)

    total_loss = 0.
    for env_outputs, env_targets in zip(outputs, targets):
        env_loss = loss_fn(env_outputs, env_targets)
        total_loss += env_loss
        env_grads = torch.autograd.grad(env_loss, params,
                                           retain_graph=True)
        for grads, env_grad in zip(param_gradients, env_grads):
            grads.append(env_grad)
    mean_loss = total_loss / n_envs
    assert len(param_gradients) == len(params)
    assert len(param_gradients[0]) == n_envs

 
    

    for param, grads in zip(params, param_gradients):
        
        def fun(x):

            f = np.zeros(grads[0].shape)
            for p, env_grad in zip(x, grads):
                f += p * env_grad.cpu().numpy()

            f_avg = np.zeros(grads[0].shape)
            for p, env_grad in zip(x, grads):
                f_avg += env_grad.cpu().numpy()
            
            return -np.dot(f.flatten(), f_avg.flatten())

        cons = ({'type':'eq', 'fun': lambda x: np.sum(x) - 1})

        bnds = list([(-2, 2) for i in range(n_envs)])

        res = minimize(fun, np.zeros(n_envs), method='SLSQP', bounds=bnds, constraints=cons)

        print(res)

    return mean_loss
