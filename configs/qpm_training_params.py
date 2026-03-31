class OptimizationScheduler:
    def __init__(self, config):
        self.config = config
        self.n_calls = 0

    def get_params(self):
        if self.n_calls == 0:  # First call returns Dense Params
            finetune = False
        else:  # Subsequent calls return Finetuning Params
            finetune = True

        self.n_calls += 1

        return self.config, finetune
