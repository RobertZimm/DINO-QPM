from dino_qpm.architectures.registry import is_finetune_mode_supported
from dino_qpm.finetuning.base import FinetuneResult, Finetuner
from dino_qpm.finetuning.qpm import finetune_qpm
from dino_qpm.finetuning.qsenn import finetune_qsenn
from dino_qpm.finetuning.sldd import finetune_sldd


class SLDDFinetuner(Finetuner):
    def run(self,
            model,
            train_loader,
            test_loader,
            log_dir,
            n_classes,
            seed,
            optimization_schedule,
            config: dict,
            run_number: int) -> FinetuneResult:
        beta = config["finetune"]["fdl"]
        return finetune_sldd(model,
                             train_loader,
                             test_loader,
                             log_dir,
                             n_classes,
                             seed,
                             optimization_schedule,
                             config=config,
                             beta=beta)


class QSENNFinetuner(Finetuner):
    def run(self,
            model,
            train_loader,
            test_loader,
            log_dir,
            n_classes,
            seed,
            optimization_schedule,
            config: dict,
            run_number: int) -> FinetuneResult:
        beta = config["finetune"]["fdl"]
        return finetune_qsenn(model,
                              train_loader,
                              test_loader,
                              log_dir,
                              n_classes,
                              seed,
                              optimization_schedule=optimization_schedule,
                              config=config,
                              beta=beta)


class QPMFinetuner(Finetuner):
    def run(self,
            model,
            train_loader,
            test_loader,
            log_dir,
            n_classes,
            seed,
            optimization_schedule,
            config: dict,
            run_number: int) -> FinetuneResult:
        beta = config["finetune"]["fdl"]
        beta_avg = config["finetune"].get("beta_avg", 0)
        n_per_class = config["finetune"]["n_per_class"]
        n_features = config["finetune"]["n_features"]

        return finetune_qpm(model,
                            train_loader,
                            test_loader,
                            log_dir,
                            n_classes,
                            seed,
                            beta,
                            optimization_schedule,
                            n_features,
                            n_per_class,
                            config=config,
                            run_number=run_number,
                            beta_avg=beta_avg)


FINETUNER_REGISTRY: dict[str, Finetuner] = {
    "sldd": SLDDFinetuner(),
    "qsenn": QSENNFinetuner(),
    "qpm": QPMFinetuner(),
}


def finetune(model,
             train_loader,
             test_loader,
             log_dir,
             n_classes,
             seed,
             optimization_schedule,
             config: dict,
             run_number: int) -> FinetuneResult:
    model_type = config["sldd_mode"]

    if model_type not in FINETUNER_REGISTRY:
        raise ValueError(f"Unknown Finetuning key: {model_type}")

    if not is_finetune_mode_supported(config["arch"], model_type):
        raise ValueError(
            f"Finetuning mode '{model_type}' is not supported for arch '{config['arch']}'."
        )

    return FINETUNER_REGISTRY[model_type].run(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        log_dir=log_dir,
        n_classes=n_classes,
        seed=seed,
        optimization_schedule=optimization_schedule,
        config=config,
        run_number=run_number,
    )
