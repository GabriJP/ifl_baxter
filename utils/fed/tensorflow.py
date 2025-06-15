import fcntl
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import TypedDict

import numpy as np
import wandb
from flwr.client import NumPyClient
from flwr.common import FitIns
from flwr.common import NDArrays
from flwr.common import Scalar
from typing_extensions import override
from wandb.integration.keras import WandbMetricsLogger

import models
from utils import denorm_max_min
from utils import F64_A
from utils import get_train_val
from utils import idx_split_traj
from utils import save_model_wandb
from utils import sort_samples_inv
from utils import Trajectory
from utils.fed import FedProxLoss


class EvalConfig(TypedDict):
    do_eval: bool


def calc_performance(y: F64_A, y_hat: F64_A) -> tuple[F64_A, float]:
    r2 = r_squared(y, y_hat)

    y_error = denorm_max_min(y) - denorm_max_min(y_hat)
    mae_joint = np.mean(np.abs(y_error), axis=0)

    return mae_joint, r2


def r_squared(y: F64_A, y_pred: F64_A) -> float:
    residual = np.sum(np.square(np.subtract(y, y_pred)))
    total = np.sum(np.square(np.subtract(y, np.mean(y))))
    return float(np.subtract(1.0, np.divide(residual, total)))


class BrnnClient(NumPyClient):
    def __init__(
        self,
        model: models.BrnnModel,
        train_data: tuple[Trajectory, ...],
        test_data: tuple[Trajectory, ...],
        online_cuts: int,
        online_additive: bool,
    ) -> None:
        super().__init__()
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.online_cuts = online_cuts
        self.online_additive = online_additive
        self.current_fit = 0

    def get_parameters(self, config: FitIns) -> NDArrays:  # noqa: ARG002
        return self.model.get_weights()

    def fit(
        self,
        parameters: NDArrays | None,
        config: FitIns,
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        if 0 < self.online_cuts <= self.current_fit + 1:
            raise ValueError

        epochs, current_epoch, batch_size = config["epochs"], config["current_epoch"], config["batch_size"]

        if not len(self.model.get_weights()):
            model_config = self.model.get_config()
            self.model.predict(np.zeros((batch_size, model_config["tx"], model_config["n_features"])))

        if parameters is not None and len(parameters):
            self.model.set_weights(parameters)

        if isinstance(self.model.loss, FedProxLoss):
            self.model.loss.update_weights_and_mu(config.get("proximal_mu"))

        step = 1
        x, y, long_traj = sort_samples_inv(self.train_data, self.model.config["tx"], step)

        mask_idx = idx_split_traj(long_traj, num_split=1)

        x_train, y_train, x_val, y_val = get_train_val(x, y, mask_idx[0])

        if self.online_cuts > 0:
            cut_len = len(x_train) // self.online_cuts
            cut_slice = (
                slice(0, cut_len * (self.current_fit + 1))
                if self.online_additive
                else slice(cut_len * self.current_fit, cut_len * (self.current_fit + 1))
            )
            x_train, y_train = x_train[cut_slice], y_train[cut_slice]

        callbacks = [
            WandbMetricsLogger(),
        ]

        self.model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            initial_epoch=current_epoch,
            batch_size=batch_size,
            verbose=2,
            callbacks=callbacks,
        )
        self.current_fit += 1
        return self.get_parameters(config), sum(map(len, self.train_data)), self.train_data[0].descriptor_dict()

    def evaluate(
        self,
        parameters: NDArrays | None = None,
        config: EvalConfig | None = None,
    ) -> tuple[float, int, dict[str, Scalar]]:
        if not len(self.model.get_weights()):
            model_config = self.model.get_config()
            self.model.predict(np.zeros((1024, model_config["tx"], model_config["n_features"])))

        if parameters is not None:
            self.model.set_weights(parameters)

        # If centralized or federated and commanded
        do_eval = config is None or config["do_eval"]
        if do_eval:
            x, y, _ = sort_samples_inv(self.test_data, self.model.config["tx"], step=1)
        else:
            x, y, long_traj = sort_samples_inv(self.train_data, self.model.config["tx"], step=1)
            mask_idx = idx_split_traj(long_traj, num_split=1)
            _, _, x, y = get_train_val(x, y, mask_idx[0])

        y_hat = self.model.predict(x)

        mae_joint, r2 = calc_performance(y, np.asarray(y_hat))
        metrics_dict: dict[str, float | bool] = dict(
            zip(("mae_s0", "mae_s1", "mae_e0", "mae_e1", "mae_w0", "mae_w1", "mae_w2"), mae_joint, strict=True)
        )

        metrics_dict.update(r2=r2, mae_mean=float(np.mean(mae_joint)))

        # If centralized
        if config is None:
            wandb.log(dict(agg=metrics_dict))
            save_model_wandb(self.model)

        metrics_dict["do_eval"] = do_eval

        return r2, sum(map(len, self.test_data)), metrics_dict


class LightParallelClient(BrnnClient):
    def __init__(
        self,
        model: models.BrnnModel,
        train_data: tuple[Trajectory, ...],
        test_data: tuple[Trajectory, ...],
        online_cuts: int,
        online_additive: bool,
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
                msg = f"Unlocking after {perf_counter() - start:02f} seconds"
                logging.info(msg)
                self.is_locked = False
                fcntl.flock(fd, fcntl.LOCK_UN)

    @override
    def fit(self, parameters: NDArrays | None, config: FitIns) -> tuple[NDArrays, int, dict[str, Scalar]]:
        with self.execution_exclusive_context():
            return super().fit(parameters, config)

    @override
    def evaluate(
        self, parameters: NDArrays | None = None, config: EvalConfig | None = None
    ) -> tuple[float, int, dict[str, Scalar]]:
        with self.execution_exclusive_context():
            return super().evaluate(parameters, config)
