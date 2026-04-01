import schedulefree
import torch
from CleanCodeRelease.configs.qpm_training_params import OptimizationScheduler
from CleanCodeRelease.configs.qsenn_training_params import QSENNScheduler
from torch.optim import SGD, lr_scheduler


def get_optimizer(model: torch.nn.Module,
                  schedulingClass: OptimizationScheduler | QSENNScheduler,
                  mode: str = None,
                  config=None):
    if config is None:
        config, _ = schedulingClass.get_params()

    if mode is None:
        _, finetune = schedulingClass.get_params()

        # Setting the mode to be able to access
        # the correct parameters from the config
        if finetune:
            mode = "finetune"
        else:
            mode = "dense"

    optimizer_type = config["optimizer"]

    # Get the parameters from the config
    start_lr = config[mode]["start_lr"]
    weight_decay = config[mode]["weight_decay"]
    epochs = config[mode]["epochs"]

    if "schedulefree" not in optimizer_type:
        step_lr = config[mode]["step_lr"]
        step_lr_decay = config[mode]["step_lr_decay"]

    if "sgd" in optimizer_type:
        momentum = config[mode]["momentum"]

    if mode == "finetune":
        param_list = [x for x in model.parameters()
                      if x.requires_grad]

        if optimizer_type == "sgd":
            optimizer = SGD(param_list,
                            lr=start_lr,
                            momentum=momentum,
                            weight_decay=weight_decay)

        elif optimizer_type == "adamw_schedulefree":
            optimizer = schedulefree.AdamWScheduleFree(param_list,
                                                       lr=start_lr,
                                                       weight_decay=weight_decay)

        elif optimizer_type == "sgd_schedulefree":
            optimizer = schedulefree.SGDScheduleFree(param_list,
                                                     lr=start_lr,
                                                     momentum=momentum,
                                                     weight_decay=weight_decay
                                                     )

        else:
            raise ValueError(f"Optimizer {optimizer_type} not supported")

    else:
        classifier_params_name = ["linear.bias",
                                  "linear.weight"]

        classifier_params = [x[1] for x in
                             list(filter(lambda kv: kv[0]
                                         in classifier_params_name,
                                         model.named_parameters()))]

        base_params = [x[1] for x in list(filter(lambda kv:
                                                 kv[0] not in
                                                 classifier_params_name,
                                                 model.named_parameters()))]

        if optimizer_type == "sgd":
            optimizer = SGD([{'params': base_params},
                             {"params": classifier_params,
                              'lr': start_lr}],
                            momentum=momentum,
                            lr=start_lr,
                            weight_decay=weight_decay)

        elif optimizer_type == "adamw_schedulefree":
            optimizer = schedulefree.AdamWScheduleFree([{'params': base_params},
                                                        {"params": classifier_params,
                                                         'lr': start_lr}],
                                                       lr=start_lr,
                                                       weight_decay=weight_decay
                                                       )

        elif optimizer_type == "sgd_schedulefree":
            optimizer = schedulefree.SGDScheduleFree([{'params': base_params},
                                                      {"params": classifier_params,
                                                       'lr': start_lr}],
                                                     lr=start_lr,
                                                     momentum=momentum,
                                                     weight_decay=weight_decay)

    # Make schedule
    if "schedulefree" not in optimizer_type:
        schedule = lr_scheduler.StepLR(optimizer,
                                       step_size=step_lr,
                                       gamma=step_lr_decay)
    else:
        schedule = None

    return optimizer, schedule, epochs


def get_scheduler_for_model(model_type,
                            config):
    if model_type == "qsenn":
        return QSENNScheduler(config)

    else:
        return OptimizationScheduler(config)
