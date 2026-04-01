from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any

import torch


@dataclass
class FinetuneResult:
    model: torch.nn.Module
    metrics: dict[str, Any] = field(default_factory=dict)


class Finetuner(ABC):
    @abstractmethod
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
        raise NotImplementedError
