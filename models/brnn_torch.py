import logging
from collections import OrderedDict
from typing import Any
from typing import TypedDict

import numpy as np
import torch
import tqdm
import wandb
from flwr.common import NDArrays

from utils import F64_A
from utils.losses import BaseLoss

txt_logger = logging.getLogger("txt_logger")


def add_tensor_dicts(log_dict: dict[str, float], update_dict: dict[str, float]) -> None:
    for k, v in update_dict.items():
        log_dict[k] += v


class BrnnTorchConfig(TypedDict):
    n_a: int
    n_values: int
    tx: int
    n_features: int


class BrnnTorch(torch.nn.Module):
    def __init__(
        self,
        loss: BaseLoss,
        n_a: int = 64,
        n_values: int = 7,
        tx: int = 25,
        n_features: int = 14,
    ) -> None:
        if tx % 2 == 0:
            msg = "Tx must be odd."
            raise ValueError(msg)

        super().__init__()
        self.config: BrnnTorchConfig = dict(n_a=n_a, n_values=n_values, tx=tx, n_features=n_features)

        self.mid_pos = tx // 2

        self.rnn_for = torch.nn.GRU(n_features, n_a, batch_first=True)
        self.rnn_back = torch.nn.GRU(n_features, n_a, batch_first=True)

        self.dense = torch.nn.Linear(n_a * 2, n_values)

        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.loss_metric = loss

    def get_weights(self) -> NDArrays:
        return [val.cpu().numpy() for val in self.state_dict().values()]

    def set_weights(self, weights: NDArrays) -> None:
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights, strict=False)})
        self.load_state_dict(state_dict, strict=True)

    def save_weights(self, save_path: str, **kwargs: Any) -> None:
        torch.save(self.state_dict(), save_path, **kwargs)

    def load_weights(self, save_path: str, **kwargs: Any) -> None:
        state_dict = torch.load(save_path, **kwargs)
        self.load_state_dict(state_dict, strict=True)

    def get_config(self) -> BrnnTorchConfig:
        return self.config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # batch_size, 13, 14
        x_start2mid = x[..., : -self.mid_pos, :]
        x_mid2end = x[..., self.mid_pos :, :]

        # batch_size, 13, 64
        forw, _ = self.rnn_for(x_start2mid)
        back, _ = self.rnn_back(torch.flip(x_mid2end, [1]))

        # batch_size, 128
        concat = torch.cat([forw[..., -1, :], back[..., -1, :]], dim=1)

        # batch_size, 7
        return self.dense(concat)

    @torch.no_grad()
    def step_call(self, x: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
        return torch.cat([self(chunk) for chunk in torch.split(x, batch_size)])

    def fit(
        self,
        x_train: F64_A,
        y_train: F64_A,
        validation_data: tuple[F64_A, F64_A],
        epochs: int,
        current_epoch: int,
        batch_size: int,
        mu: float,
    ) -> None:
        if len(x_train) != len(y_train) or len(validation_data[0]) != len(validation_data[1]):
            raise ValueError

        global_params = [param.detach().clone() for param in self.parameters(recurse=True)]

        x_train_t = torch.from_numpy(x_train.astype(np.float32)).to(device=self.device, dtype=torch.float32)
        y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(device=self.device, dtype=torch.float32)
        x_valid_t = torch.from_numpy(validation_data[0].astype(np.float32)).to(device=self.device, dtype=torch.float32)
        y_valid_t = torch.from_numpy(validation_data[1].astype(np.float32)).to(device=self.device, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, eps=1e-7)

        for epoch in range(current_epoch, epochs):
            self.train()
            self.loss_metric.clear()

            perm = torch.randperm(len(x_train_t), device=self.device)
            x_train_split = torch.split(x_train_t[perm], batch_size)
            y_train_split = torch.split(y_train_t[perm], batch_size)

            for step_x_train, step_y_train in tqdm.tqdm(
                zip(x_train_split, y_train_split, strict=True),
                total=len(x_train_split),
                desc=f"Epoch {epoch + 1}/{epochs}",
                leave=True,
                unit="step",
            ):
                optimizer.zero_grad()

                outputs = self(step_x_train)

                # For velocity loss: select batch data, the 13th point is the one corresponding to the current
                # prediction, then select only velocities
                loss = self.loss_metric(outputs, step_y_train)

                if mu > 0.0:
                    proximal_term = sum(
                        (local_weights - global_weights).norm(2)
                        for local_weights, global_weights in zip(
                            self.parameters(recurse=True), global_params, strict=False
                        )
                    )
                    loss += mu / 2 * proximal_term
                    self.loss_metric.log_data("fedprox_loss", loss, loss)

                loss.backward()
                optimizer.step()

            wandb_dict = self.loss_metric.log(mean_factor=len(y_train_t))
            self.loss_metric.clear()

            self.eval()
            with torch.no_grad():
                outputs = self.step_call(x_valid_t, batch_size=batch_size)
                self.loss_metric(outputs, y_valid_t)

            wandb_dict.update(self.loss_metric.log("val_", mean_factor=len(y_valid_t)))
            wandb.log(wandb_dict)
            txt_logger.info(wandb_dict)

    @torch.no_grad()
    def predict(self, x: F64_A) -> torch.Tensor:
        self.eval()
        x_t = torch.from_numpy(x).to(device=self.device, dtype=torch.float32)

        return self.step_call(x_t)
