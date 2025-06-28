import fcntl
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import TypedDict

import numpy as np
import tensorflow as tf
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


def calc_performance(y: F64_A, y_hat: tf.Tensor) -> tuple[F64_A, tf.Tensor]:
    r2 = r_squared(tf.convert_to_tensor(y), y_hat)

    y_error = denorm_max_min(y) - denorm_max_min(np.zeros(shape=tuple(y_hat.shape)) + y_hat)
    mae_joint = np.mean(np.abs(y_error), axis=0)

    return mae_joint, r2


def r_squared(y: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    return tf.subtract(1.0, tf.divide(residual, total))


def aggregate_paths_descriptors(trajectories: tuple[Trajectory, ...]) -> dict[str, Scalar]:
    result_dict: dict[str, Scalar] = dict(names="")
    for trajectory in trajectories:
        descriptor_dict = trajectory.descriptor_dict()
        for pos_name, pos_value in descriptor_dict.items():
            result_dict[f"{trajectory.name}_{pos_name}"] = pos_value
            result_dict["names"] += f"/{trajectory.name}"
    return result_dict


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

    def get_train_data(self) -> tuple[F64_A, F64_A, F64_A, F64_A]:
        train_data = list(self.train_data)

        # If continuous learning and more cuts available, cut data
        if 0 < self.online_cuts <= self.current_fit:
            train_data = list()
            for traj in self.train_data:
                cut_len = len(traj.path) // self.online_cuts
                cut_slice = slice(
                    # Select from the beginning of (whole set or current cut)
                    0 if self.online_additive else cut_len * self.current_fit,
                    # Do not go out of bounds
                    min(len(traj.path), cut_len * (self.current_fit + 1)),
                )
                train_data.append(Trajectory(traj.name, traj.descriptor, traj.path[cut_slice]))

        x, y, long_traj = sort_samples_inv(train_data, self.model.config["tx"], step=1)

        mask_idx = idx_split_traj(long_traj, num_split=1)

        x_train, y_train, x_val, y_val = get_train_val(x, y, mask_idx[0])
        return x_train, y_train, x_val, y_val

    def fit(
        self,
        parameters: NDArrays | None,
        config: FitIns,
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        if parameters is not None:
            self.model.set_weights(parameters)

        if isinstance(self.model.loss, FedProxLoss):
            self.model.loss.update_initial_weights_and_mu(self.model, config.get("proximal_mu"))

        epochs, current_epoch, batch_size = config["epochs"], config["current_epoch"], config["batch_size"]

        x_train, y_train, x_val, y_val = self.get_train_data()

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
        return self.get_parameters(config), sum(map(len, self.train_data)), aggregate_paths_descriptors(self.train_data)

    def evaluate(
        self,
        parameters: NDArrays | None = None,
        config: EvalConfig | None = None,
    ) -> tuple[float, int, dict[str, Scalar]]:
        if parameters is not None:
            self.model.set_weights(parameters)

        # If centralized or federated and commanded
        do_eval = config is None or config["do_eval"]
        if do_eval:
            x, y, _ = sort_samples_inv(self.test_data, self.model.config["tx"], step=1)
        else:
            _, _, x, y = self.get_train_data()

        y_hat = self.model.predict(x)

        mae_joint, r2 = calc_performance(y, y_hat)
        metrics_dict = dict(
            zip(("mae_s0", "mae_s1", "mae_e0", "mae_e1", "mae_w0", "mae_w1", "mae_w2"), mae_joint, strict=True)
        )
        r2_float = float(r2.numpy())

        metrics_dict.update(r2=r2_float, mae_mean=float(np.mean(mae_joint)))

        # If centralized
        if config is None:
            wandb.log(dict(agg=metrics_dict))
            save_model_wandb(self.model)

        metrics_dict["do_eval"] = do_eval

        return r2_float, sum(map(len, self.test_data)), metrics_dict


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
