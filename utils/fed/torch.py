import fcntl
import logging
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from time import sleep
from typing import NotRequired
from typing import override
from typing import TypedDict

import numpy as np
import torch
import wandb
from flwr.client import NumPyClient
from flwr.common import NDArrays
from flwr.common import Scalar

from models import BrnnTorch
from utils import denorm_max_min
from utils import F64_A
from utils import get_train_val
from utils import idx_split_traj
from utils import sort_samples_inv
from utils import Trajectory

txt_logger = logging.getLogger("txt_logger")


@torch.no_grad()
def r_squared(y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    residual = torch.sum(torch.square(torch.subtract(y, y_pred)))
    total = torch.sum(torch.square(torch.subtract(y, torch.mean(y))))
    return torch.subtract(torch.tensor(1.0, dtype=total.dtype), torch.divide(residual, total))


def calc_performance(y: torch.Tensor, y_hat: torch.Tensor) -> tuple[F64_A, float]:
    r2 = float(r_squared(y, y_hat).cpu().numpy())

    y_error = denorm_max_min(y.cpu().numpy()) - denorm_max_min(y_hat.cpu().numpy())
    mae_joint = np.mean(np.abs(y_error), axis=0)

    return mae_joint, r2


class FitConfig(TypedDict):
    epochs: int
    current_epoch: int
    batch_size: int
    proximal_mu: NotRequired[float]


class EvalConfig(TypedDict):
    do_eval: bool


class TorchClient(NumPyClient):
    def __init__(
        self,
        model: BrnnTorch,
        train_data: tuple[Trajectory, ...],
        test_data: tuple[Trajectory, ...],
        online_cuts: int,
        online_additive: int,
    ) -> None:
        self.model: BrnnTorch = model
        if torch.cuda.is_available():
            self.model = model.to("cuda")
            # self.model.compile(dynamic=False, mode="reduce-overhead")
        self.train_data = train_data
        self.test_data = test_data
        self.online_cuts = online_cuts
        self.online_additive = online_additive

    def get_parameters(self, _: FitConfig) -> NDArrays:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def fit(self, parameters: NDArrays | None, config: FitConfig) -> tuple[NDArrays, int, dict[str, Scalar]]:
        if parameters is not None:
            state_dict = OrderedDict(
                {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters, strict=False)}
            )
            self.model.load_state_dict(state_dict, strict=True)

        epochs, current_epoch, batch_size, mu = (
            config["epochs"],
            config["current_epoch"],
            config["batch_size"],
            config.get("proximal_mu", 0.0),
        )

        step = 1
        x, y, long_traj = sort_samples_inv(self.train_data, self.model.config["tx"], step)

        mask_idx = idx_split_traj(long_traj, num_split=1)

        x_train, y_train, x_val, y_val = get_train_val(x, y, mask_idx[0])

        self.model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            current_epoch=current_epoch,
            batch_size=batch_size,
            mu=mu,
        )

        torch.cuda.empty_cache()

        return self.get_parameters(config), len(self.train_data), self.train_data[0].descriptor_dict()

    @torch.no_grad()
    def evaluate(
        self, parameters: NDArrays | None = None, config: EvalConfig | None = None
    ) -> tuple[float, int, dict[str, Scalar]]:
        # If federated
        if parameters is not None:
            self.model.set_weights(parameters)

        # If centralized or federated and commanded
        if config is None or config["do_eval"]:
            x, y, _ = sort_samples_inv(self.test_data, self.model.config["tx"], step=1)
        else:
            x, y, long_traj = sort_samples_inv(self.train_data, self.model.config["tx"], step=1)
            mask_idx = idx_split_traj(long_traj, num_split=1)
            _, _, x, y = get_train_val(x, y, mask_idx[0])

        y_t = torch.from_numpy(y).to(device=self.model.device, dtype=torch.float32)

        y_hat = self.model.predict(x)

        mae_joint, r2 = calc_performance(y_t, y_hat)
        mae_dict = dict(mae_mean=float(np.mean(mae_joint)))
        mae_dict.update(
            zip(
                ("mae_s0", "mae_s1", "mae_e0", "mae_e1", "mae_w0", "mae_w1", "mae_w2"),
                mae_joint.tolist(),
                strict=True,
            )
        )

        wandb_dict = dict(agg=dict(r2=r2, **mae_dict))
        if config is None or config["do_eval"]:
            wandb.log(wandb_dict)
            txt_logger.info(wandb_dict)
        return r2, sum(map(len, self.test_data)), mae_dict


class LightParallelClient(TorchClient):
    def __init__(
        self,
        model: BrnnTorch,
        train_data: tuple[Trajectory, ...],
        test_data: tuple[Trajectory, ...],
        online_cuts: int,
        online_additive: int,
    ) -> None:
        super().__init__(model, train_data, test_data, online_cuts, online_additive)
        self.is_locked = False

    @contextmanager
    def execution_exclusive_context(self) -> Iterator[None]:
        if self.is_locked:
            logging.info("Lock skipped")
            yield
            logging.info("Unlock skipped")
            return
        with Path(__file__).open() as fd:
            start = float("inf")
            try:
                logging.info("Locking")
                fcntl.flock(fd, fcntl.LOCK_EX)
                start = perf_counter()
                self.is_locked = True
                logging.info("Locked")
                yield
            finally:
                torch.cuda.empty_cache()
                msg = f"Unlocking after {perf_counter() - start:02f} seconds"
                logging.info(msg)
                self.is_locked = False
                fcntl.flock(fd, fcntl.LOCK_UN)

    @override
    def fit(self, parameters: NDArrays | None, config: FitConfig) -> tuple[NDArrays, int, dict[str, Scalar]]:
        with self.execution_exclusive_context():
            return super().fit(parameters, config)

    @override
    def evaluate(
        self, parameters: NDArrays | None = None, config: EvalConfig | None = None
    ) -> tuple[float, int, dict[str, Scalar]]:
        with self.execution_exclusive_context():
            return super().evaluate(parameters, config)


class ParallelClient(TorchClient):
    def __init__(
        self,
        model: BrnnTorch,
        train_data: tuple[Trajectory, ...],
        test_data: tuple[Trajectory, ...],
        online_cuts: int,
        online_additive: int,
    ) -> None:
        super().__init__(model, train_data, test_data, online_cuts, online_additive)
        self.current_device = torch.device("cpu")
        self.to_cpu()
        self.is_locked = False

    def to_device(self, device: torch.device) -> None:
        self.model.to(device)
        self.current_device = device

    def to_cpu(self) -> None:
        self.to_device(torch.device("cpu"))

    def to_gpu(self) -> None:
        self.to_device(torch.device("cuda"))

    @contextmanager
    def execution_exclusive_context(self, *, to_gpu: bool = False) -> Iterator[None]:
        if self.is_locked:
            logging.info("Lock skipped")
            yield
            logging.info("Unlock skipped")
            return
        with Path(__file__).open() as fd:
            start = float("inf")
            try:
                logging.info("Locking")
                fcntl.flock(fd, fcntl.LOCK_EX)
                start = perf_counter()
                self.is_locked = True
                logging.info("Locked")
                if to_gpu:
                    self.to_gpu()
                yield
            finally:
                self.to_cpu()
                torch.cuda.empty_cache()
                sleep(1)
                msg = f"Unlocking after {perf_counter() - start:02f} seconds"
                logging.info(msg)
                self.is_locked = False
                fcntl.flock(fd, fcntl.LOCK_UN)

    @override
    def fit(self, parameters: NDArrays | None, config: FitConfig) -> tuple[NDArrays, int, dict[str, Scalar]]:
        with self.execution_exclusive_context(to_gpu=True):
            return super().fit(parameters, config)

    @override
    def evaluate(
        self, parameters: NDArrays | None = None, config: EvalConfig | None = None
    ) -> tuple[float, int, dict[str, Scalar]]:
        with self.execution_exclusive_context(to_gpu=True):
            return super().evaluate(parameters, config)
