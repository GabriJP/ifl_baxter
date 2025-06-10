from typing import Any
from typing import TYPE_CHECKING

import keras

if TYPE_CHECKING:
    from keras.src.engine.keras_tensor import KerasTensor


@keras.utils.register_keras_serializable()
class BrnnModel(keras.Model):
    def __init__(
        self,
        n_a: int = 64,
        n_values: int = 7,
        tx: int = 25,
        n_features: int = 14,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config = dict(n_a=n_a, n_values=n_values, tx=tx, n_features=n_features)

        mid_pos = tx // 2

        self.x_start2mid = keras.layers.Cropping1D((0, mid_pos))
        self.x_mid2end = keras.layers.Cropping1D((mid_pos, 0))
        self.rnn_for = keras.layers.GRU(n_a, name="rnn_for")
        self.rnn_back = keras.layers.GRU(n_a, go_backwards=True, name="rnn_back")
        self.concat = keras.layers.Concatenate()
        self.densor = keras.layers.Dense(units=n_values, name="densor_out")
        self.loss = keras.losses.MeanSquaredError()

    @property
    def default_input_shape(self) -> tuple[None, int, int]:
        return None, self.config["tx"], self.config["n_features"]

    def build(self, input_shape: tuple[None, int, int] | None = None) -> None:
        if input_shape is None:
            input_shape = self.default_input_shape

        super().build(input_shape)

    def build_graph(self, input_shape: tuple[int, int] | None = None) -> keras.Model:
        if input_shape is None:
            input_shape = self.default_input_shape[1:]

        x = keras.layers.Input(shape=input_shape)
        return keras.Model(inputs=[x], outputs=self.call(x))

    def compile(self, **kwargs: Any) -> None:
        self.loss = keras.losses.MeanSquaredError()

        fedprox_pmu = kwargs.pop("fedprox_pmu", 0.0)
        if fedprox_pmu > 0.0:
            from utils.fed import FedProxLoss

            self.loss = FedProxLoss(self.loss, self)

        opts = dict(
            loss=self.loss,
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
        )
        opts.update(kwargs)

        super().compile(**opts)

    def get_config(self) -> dict[str, Any]:
        return super().get_config() | self.config

    def call(self, inputs: Any, **_: Any) -> "KerasTensor":
        # Split data
        # None, 25, 14 -> None, 13, 14
        forw = self.x_start2mid(inputs)
        back = self.x_mid2end(inputs)

        # Recurrent layers
        forw = self.rnn_for(forw)
        back = self.rnn_back(back)

        # Concatenate recurrent output
        conc = self.concat([forw, back])

        # Dense output
        return self.densor(conc)
