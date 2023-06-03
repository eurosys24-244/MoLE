from torch.optim.adamw import adamw
from torch.optim.adam import adam 
from torch.optim.adagrad import adagrad
import torch_optimizer
import lion_pytorch 
import pytorch_lamb
import torch
import copy 
class OptRegister(dict):
    def __init__(self, *args, **kwargs):
        super(OptRegister, self).__init__(*args, **kwargs)
        self._dict = {}

    def __call__(self, target):
        return self.register(target)

    def register(self, target):
        def add_item(key, value):
            if not callable(value):
                raise Exception(f"Error:{value} must be callable!")
            if key in self._dict:
                print(f"Warning: {value.__name__} already exists and will be overwritten!")
            self[key] = value
            return value

        if callable(target):    
            return add_item(target.__name__[:target.__name__.find("_")], target)
        else:                   
            return lambda x : add_item(target, x)

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __str__(self):
        return str(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

opt_register = OptRegister()
OtherStates = {}
MovableStates = {}
CustomOptimizer = {}

def other_states_register(opt_name, lst_states):
    OtherStates[opt_name] = lst_states

def movable_states_register(opt_name, lst_states):
    MovableStates[opt_name] = lst_states

def custom_optimizer_register(opt_name, opt):
    CustomOptimizer[opt_name] = opt

@opt_register
def adagrad_helper(param, grad, opt_state, non_opt_state, group):



    adagrad(
        params=[param],
        grads=[grad],
                state_sums=[opt_state[0]],
                state_steps=[non_opt_state[0]],
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                lr_decay=group["lr_decay"],
                eps=group["eps"],
                has_sparse_grad=False,
                foreach=group["foreach"],
                maximize=group["maximize"],
                differentiable=group["differentiable"],
            )


@opt_register
def adamw_helper(param, grad,opt_state,non_opt_state,group):

    if type(non_opt_state[0]) == int:
        non_opt_state[0] = torch.tensor(non_opt_state[0])
    adamw(
    params=[param],
    grads=[grad],
    exp_avgs=[opt_state[0]],
    exp_avg_sqs=[opt_state[1]],
    max_exp_avg_sqs=[None],
    state_steps=[non_opt_state[0]],
    beta1 = group["betas"][0],
    beta2 = group["betas"][1],
    lr=group["lr"],
    weight_decay=group["weight_decay"],
    eps=group["eps"],
    maximize=False,
    amsgrad=group["amsgrad"],
    )


@opt_register
def lamb_helper(param, grad, opt_state, non_opt_state, group):
    p = param
    exp_avg, exp_avg_sq = opt_state[0], opt_state[1]
    beta1, beta2 = group['betas']

    non_opt_state[0] += 1 #useless adding here

    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    step_size = group['lr'] # * math.sqrt(bias_correction2) / bias_correction1

    weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

    adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
    if group['weight_decay'] != 0:
        adam_step.add_(p.data, alpha=group['weight_decay'])

    adam_norm = adam_step.pow(2).sum().sqrt()
    if weight_norm == 0 or adam_norm == 0:
        trust_ratio = 1
    else:
        trust_ratio = weight_norm / adam_norm

    p.data.add_(adam_step, alpha=-step_size * trust_ratio)

@opt_register
def lion_helper(param, grad, opt_state, non_opt_state, group):
    p, grad, exp_avg, lr, wd, beta1, beta2 = param, grad, opt_state[0], group["lr"], group["weight_decay"], group["betas"][0], group["betas"][1]
    lion_pytorch.lion_pytorch.update_fn(p, grad, exp_avg, lr, wd, beta1, beta2)

movable_states_register("adamw", ["exp_avg", "exp_avg_sq"])

other_states_register("adamw", ["step"])

custom_optimizer_register("adamw", torch.optim.AdamW)

movable_states_register("adagrad", ["sum"])

other_states_register("adagrad", ["step"])

custom_optimizer_register("adagrad", torch.optim.Adagrad)

movable_states_register("sgd", [])

other_states_register("sgd", ["step"])

custom_optimizer_register("sgd", torch.optim.SGD)

movable_states_register("lamb", ["exp_avg", "exp_avg_sq"])

other_states_register("lamb", ["step"])

custom_optimizer_register("lamb", torch_optimizer.Lamb)

movable_states_register("lion", ["exp_avg"])

other_states_register("lion", [])

custom_optimizer_register("lion", lion_pytorch.Lion)

def get_helper(opt_name):
    return opt_register[opt_name]

def get_movable_states(opt_name):
    return MovableStates[opt_name] 

def get_other_states(opt_name):
    return OtherStates[opt_name]

def get_custom_optimizer(opt_name, cpu=False):
    if opt_name == "adamw" and cpu:
        import deepspeed 
        return deepspeed.ops.adam.DeepSpeedCPUAdam
    return CustomOptimizer[opt_name]