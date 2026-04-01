from dino_qpm.configs.core.qpm_training_params import OptimizationScheduler


class QSENNScheduler(OptimizationScheduler):
    def get_params(self):
        config, finetune = super().get_params()

        if finetune:
            mode = "finetune"
        else:
            mode = "dense"

        if self.n_calls >= 2:
            config[mode]["start_lr"] *= 0.9**(self.n_calls-2)

        if 2 <= self.n_calls <= 4:
            # Change num epochs to 10 for iterative finetuning
            config[mode]["epochs"] = 10

        return config, finetune
