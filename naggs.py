# This file could be used with general PyTorch pipeline
import torch as T

from typing import List, Optional

__all__ = ('NagGS', 'nag_gs')

class NagGS(T.optim.Optimizer):
    """Class NagGS implements algorithm with update on gamma which stands for
    the semi-implicit integration of the Nesterov Accelerated Gradient (NAG)
    flow.
    Arguments
    ---------
        params (collection): Collection of parameters to optimize.
        lr (float, optional): Learning rate (or alpha).
        mu (float, optional): Momentum mu.
        gamm (float, optional): Gamma factor.
    """

    def __init__(self, params, lr=1e-2, mu=1.0, gamma=1.0):
        if lr < 0.0:
            raise ValueError(f"alpha: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid mu: {mu}")
        if gamma < 0.0:
            raise ValueError('Parameter gamma should be non-nevative.')

        defaults = dict(lr=lr, mu=mu, gamma=gamma)
        super().__init__(params, defaults)

    @T.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments
        ---------
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with T.enable_grad():
                loss = closure()

        for group in self.param_groups:
            state = []
            params = []
            grads = []
            gammas = []

            for p in filter(lambda x: x.grad is not None, group['params']):
                if len(param_state := self.state[p]) == 0:
                    param_state['state'] = T.clone(p).detach()
                state.append(param_state['state'])
                params.append(p)
                grads.append(p.grad)
                gammas.append(param_state.get('gamma', group['gamma']))

            nag_gs(state, params, grads, group['lr'], gammas, group['mu'])

            for param, gamma in zip(params, gammas):
                self.state[param]['gamma'] = gamma

        return loss


def nag_gs(state: List[T.Tensor], params: List[T.Tensor],
           grads: List[T.Tensor], alpha: float, gammas: List[float],
           mu: float):
    """Function nag-gs performs full NAG-GS algorithm computation.
    """
    gammas_out = []
    for gamma, gs, xs, vs in zip(gammas, grads, params, state):
        # Update constants
        a = alpha / (alpha + 1)
        b = alpha * mu / (alpha * mu + gamma)
        gamma = (1-a)*gamma + a*mu
        gammas_out.append(gamma)
        
        # Update state v
        vs.mul_(1 - b)
        vs.add_(xs, alpha=b)
        vs.add_(gs, alpha=-b / mu)
        
        # Update parameters x
        xs.mul_(1 - a)
        xs.add_(vs, alpha=a)

    return gammas_out