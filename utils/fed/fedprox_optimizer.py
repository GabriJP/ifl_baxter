import logging
from typing import Final
from typing import TYPE_CHECKING

import keras
from keras import losses

if TYPE_CHECKING:
    from keras.src.engine.keras_tensor import KerasTensor


txt_logger: Final = logging.getLogger("txt_logger")


class FedProxLoss(losses.Loss):
    def __init__(self, base_loss: losses.Loss, model: keras.Model) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.model = model
        self.proximal_mu_half = 0.0
        self.initial_params: list[KerasTensor] = list()

    def update_weights_and_mu(self, proximal_mu: float) -> None:
        self.initial_params = self.model.get_weights()
        self.proximal_mu_half = proximal_mu / 2

    def call(self, y_true: "KerasTensor", y_pred: "KerasTensor") -> "KerasTensor":
        loss = self.base_loss(y_true, y_pred)

        proximal_loss = 0.0
        current_weights = self.model.get_weights()
        for initial_weight, current_weight in zip(self.initial_params, current_weights, strict=False):
            proximal_loss += keras.ops.sqrt(keras.ops.sum(keras.ops.square(initial_weight - current_weight)))

        txt_logget_msg = f"FedProxLoss. loss: {loss}, proximal_loss: {proximal_loss}"
        txt_logger.info(txt_logget_msg)

        return loss + self.proximal_mu_half * proximal_loss
