from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from typing import Any

import torch

STEP_INDEX = 4


class BaseLoss(ABC, torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.latest_data_log: dict[str, torch.Tensor] = dict()
        self.sum_log_data: dict[str, torch.Tensor] = defaultdict(
            lambda: torch.tensor(0.0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        )

    @abstractmethod
    def forward(self, output: torch.Tensor, target: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor: ...

    def clear(self) -> None:
        self.latest_data_log.clear()
        self.sum_log_data.clear()

    def log(self, prefix: str = "", mean_factor: int = 1) -> dict[str, float]:
        return {f"{prefix}{k}": v.cpu().item() / mean_factor for k, v in self.sum_log_data.items()}

    def log_data(self, key: str, value: torch.Tensor, sum_value: torch.Tensor, *, sum_calc: bool = True) -> None:
        self.latest_data_log[key] = value
        if sum_calc:
            self.sum_log_data[key] += sum_value
        else:
            self.sum_log_data[key] = sum_value


class MSELoss(BaseLoss):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.loss_fn = torch.nn.MSELoss(reduction="none")

    def _mse_der_helper(self, output: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mse_losses = self.loss_fn(output, target)
        der_losses = self.loss_fn(output[1:], output[:-1])

        mse_loss = torch.mean(mse_losses)
        der_loss = torch.mean(der_losses)

        self.log_data("mse", mse_loss, torch.sum(mse_losses))
        self.log_data("der_loss", der_loss, torch.sum(der_losses))

        return mse_loss, der_loss

    def forward(self, output: torch.Tensor, target: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        mse_loss, _ = self._mse_der_helper(output, target)  # type: ignore[assignment]
        self.log_data("loss", mse_loss, self.sum_log_data["mse"], sum_calc=False)
        return mse_loss


class StepLoss(MSELoss):
    def forward(self, output: torch.Tensor, target: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        return super().forward(output[:, :STEP_INDEX], target[:, :STEP_INDEX], **kwargs)


class CVLoss(MSELoss):
    def __init__(self, cv_loss: float, n_values: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        value_at_0 = cv_loss
        slope = (1 - value_at_0 - value_at_0) / (n_values - 1)
        self.aux_tensor = torch.tensor(
            [i * slope + value_at_0 + 0.5 for i in range(int(n_values))],
            dtype=torch.float32,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    def forward(self, output: torch.Tensor, target: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        self._mse_der_helper(output, target)

        mse_loss = self.loss_fn(output, target)
        mse_min, mse_max = torch.min(mse_loss), torch.max(mse_loss)
        mse_loss = torch.mean(torch.mean(mse_loss, dim=0) * self.aux_tensor)
        mse_loss = (mse_loss - mse_min) / (mse_max - mse_min)

        self.latest_data_log["loss"] = mse_loss

        return mse_loss


class VelocityLoss(MSELoss):
    def forward(self, output: torch.Tensor, target: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        if "vel" not in kwargs:
            raise ValueError

        self._mse_der_helper(output, target)

        vel_loss = self.loss_fn(output, target) * kwargs["vel"]
        vel_mse_loss = torch.mean(vel_loss)

        self.log_data("loss", vel_mse_loss, torch.sum(vel_loss))

        return vel_mse_loss


class MSEDerivativeLoss(MSELoss):
    def __init__(self, lambda_: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lambda_ = torch.tensor(
            lambda_, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.lambda_op = torch.tensor(
            1 - lambda_, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def forward(self, output: torch.Tensor, target: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        mse_loss, der_loss = self._mse_der_helper(output, target)
        loss = self.lambda_ * mse_loss + self.lambda_op * der_loss

        mse_losses = torch.sum(self.loss_fn(output, target))
        der_losses = torch.sum(self.loss_fn(output[1:], output[:-1]))

        self.log_data("loss", loss, self.lambda_ * mse_losses + self.lambda_op * der_losses)

        return loss


class VelMSEDerivativeLoss(MSEDerivativeLoss):
    def forward(self, output: torch.Tensor, target: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        if "vel" not in kwargs:
            raise ValueError

        _, der_loss = self._mse_der_helper(output, target)

        vel_loss = torch.mean(self.loss_fn(output, target) * kwargs["vel"])

        loss = self.lambda_ * vel_loss + self.lambda_op * der_loss

        mse_losses = torch.sum(self.loss_fn(output, target))
        der_losses = torch.sum(self.loss_fn(output[1:], output[:-1]))

        self.log_data("loss", loss, self.lambda_ * mse_losses + self.lambda_op * der_losses)

        return loss


class StepVDerLoss(VelMSEDerivativeLoss):
    def forward(self, output: torch.Tensor, target: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        if "vel" not in kwargs:
            raise ValueError

        return super().forward(
            output[:, :STEP_INDEX], target[:, :STEP_INDEX], vel=kwargs.pop("vel")[:, :STEP_INDEX], **kwargs
        )


LOSSES = dict(
    cv=CVLoss,
    der=MSEDerivativeLoss,
    mse=MSELoss,
    step=StepLoss,
    stepvder=StepVDerLoss,
    vder=VelMSEDerivativeLoss,
    vel=VelocityLoss,
)
