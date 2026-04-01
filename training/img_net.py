from torch.optim import SGD, lr_scheduler


def get_default_img_optimizer(model):
    return SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    raise NotImplementedError("TODO: Implement get_default_img_optimizer")
    pass

def get_default_img_schedule(default_img_optimizer):
    return lr_scheduler.StepLR(default_img_optimizer, step_size=30, gamma=0.1)
    raise NotImplementedError("TODO: Implement get_default_img_schedule")
    pass