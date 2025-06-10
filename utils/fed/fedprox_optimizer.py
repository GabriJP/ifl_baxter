import keras
import tensorflow as tf
from keras import losses


class FedProxLoss(losses.Loss):
    def __init__(self, base_loss: losses.Loss, model: keras.Model) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.model = model
        self.proximal_mu_half = 0.0
        self.initial_params: list[tf.Tensor] = list()

    def update_initial_weights_and_mu(self, model: keras.Model, proximal_mu: float) -> None:
        self.initial_params = model.get_weights()
        self.proximal_mu_half = proximal_mu / 2

    @tf.function  # type: ignore[misc]
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        loss = self.base_loss(y_true, y_pred)

        proximal_loss = 0.0
        current_weights = self.model.get_weights()
        for initial_weight, current_weight in zip(self.initial_params, current_weights, strict=False):
            proximal_loss += tf.norm(initial_weight - current_weight, ord=2)

        print(f"FedProxLoss. loss: {loss}, proximal_loss: {proximal_loss}")

        return loss + self.proximal_mu_half * proximal_loss
