import itertools
from collections.abc import Callable
from functools import reduce
from logging import WARNING

import numpy as np
import numpy.typing as npt
from flwr.common import FitRes
from flwr.common import log
from flwr.common import MetricsAggregationFn
from flwr.common import NDArrays
from flwr.common import ndarrays_to_parameters
from flwr.common import Parameters
from flwr.common import parameters_to_ndarrays
from flwr.common import Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

F64_A = npt.NDArray[np.float64]
COORDINATE = tuple[float, float, float]


class Cube:
    def __init__(self, descriptor: F64_A):
        """
        Class representing a cube.

        :param descriptor: Array of shape (2, 3) representing the two coordinates for the cube
        """
        if descriptor.shape != (2, 3):
            msg = "Points must have exactly 3 points"
            raise ValueError(msg)

        self.descriptor: F64_A = np.stack([descriptor.min(axis=0), descriptor.max(axis=0)], dtype=np.float64)

        if self.volume <= 0.0:
            msg = "Invalid points"
            raise ValueError(msg)

    def __hash__(self) -> int:
        return hash(str(self.descriptor))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Cube) and (self.descriptor == other.descriptor).all()

    @classmethod
    def from_tuples(cls, point_a: COORDINATE, point_b: COORDINATE) -> "Cube":
        return cls(np.stack([np.array(point_a), np.array(point_b)]))

    @classmethod
    def from_dict(cls, pos_dict: dict[str, float]) -> "Cube":
        a_0, a_1, a_2 = pos_dict.pop("a_0"), pos_dict.pop("a_1"), pos_dict.pop("a_2")
        b_0, b_1, b_2 = pos_dict.pop("b_0"), pos_dict.pop("b_1"), pos_dict.pop("b_2")

        return cls(np.array([[a_0, a_1, a_2], [b_0, b_1, b_2]], dtype=np.float64))

    @property
    def coordinates(self) -> tuple[tuple[float, ...], ...]:
        point_a = self.descriptor[0]
        point_b = self.descriptor[1]
        all_points = (
            [point_a, point_b]
            + [[*point_a[:i], b, *point_a[i + 1 :]] for i, b in enumerate(point_b)]
            + [[*point_b[:i], a, *point_b[i + 1 :]] for i, a in enumerate(point_a)]
        )
        return tuple(map(tuple, all_points))

    def intersects(self, other: "Cube") -> bool:
        return all(b > a for b, a in zip(self.descriptor[1], other.descriptor[0], strict=False)) and all(
            a < b for a, b in zip(self.descriptor[0], other.descriptor[1], strict=False)
        )

    def intersection_cube(self, other: "Cube") -> "Cube":
        if not self.intersects(other):
            msg = "There is no intersection"
            raise ValueError(msg)
        return Cube(
            np.stack(
                [
                    np.stack([self.descriptor[0], other.descriptor[0]]).max(axis=0),
                    np.stack([self.descriptor[1], other.descriptor[1]]).min(axis=0),
                ]
            )
        )

    @property
    def volume(self) -> float:
        return np.prod(self.descriptor[1] - self.descriptor[0]).item()


def deaggregate_path_descriptors(fr: FitRes) -> list[Cube]:
    all_names = fr.metrics["names"].split("/")[1:]
    return [Cube.from_dict({k[-3:]: v for k, v in fr.metrics.items() if k[:-4] == name}) for name in all_names]


def aggregate(results: list[tuple[NDArrays, int, list[Cube]]], q: bool) -> NDArrays:
    """Compute weighted average."""
    # Calculate intersection cube for each trajectory
    traj_cubes: list[list[Cube]] = [c for _, _, c in results]
    flatten_traj_cubes = [c for lc in traj_cubes for c in lc]
    intersections: dict[Cube, list[Cube]] = {c: list() for c in flatten_traj_cubes}
    for c1, c2 in itertools.combinations(flatten_traj_cubes, 2):
        if not c1.intersects(c2):
            continue
        intersection_cube = c1.intersection_cube(c2)
        intersections[c1].append(intersection_cube)
        intersections[c2].append(intersection_cube)

    # Calculate weight for each trajectory
    traj_weights = {
        cube: -sum((i.volume for i in intersections), start=0.0) / cube.volume
        for cube, intersections in intersections.items()
    }

    min_weight = min(traj_weights.values(), default=0.0)
    max_weight = max(traj_weights.values(), default=0.0)
    # No intersections = do not apply weight. Weight ∈ (-∞, 0.0], weight is 0.0 when there is no intersection
    # Also if all overlap the same amount
    if np.isclose(min_weight, max_weight):
        norm_traj_weights = {c: 1.0 for c in traj_weights}
    else:
        ptp_weight = (max_weight - min_weight) * 9
        norm_traj_weights = {c: (w - min_weight) / ptp_weight + 0.1 for c, w in traj_weights.items()}

    def sum_norm_traj_weights(cubes: list[Cube]) -> float:
        return sum(norm_traj_weights[cube] for cube in cubes)

    # Create a list of weights, each multiplied by the related weight
    weighted_weights = [
        [layer * (num_examples if q else 1.0) * sum_norm_traj_weights(cubes) for layer in weights]
        for weights, num_examples, cubes in results
    ]
    total_divide = sum(
        (num_examples if q else 1.0) * sum_norm_traj_weights(cubes) for _, num_examples, cubes in results
    )

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / total_divide for layer_updates in zip(*weighted_weights, strict=False)
    ]
    return weights_prime


class CubeStrategy(FedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: (
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                tuple[float, dict[str, Scalar]] | None,
            ]
            | None
        ) = None,
        on_fit_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        on_evaluate_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        accept_failures: bool = True,
        initial_parameters: Parameters | None = None,
        fit_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        q: bool,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.q = q

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, deaggregate_path_descriptors(fit_res))
            for _, fit_res in results
        ]
        aggregated_ndarrays = aggregate(weights_results, self.q)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
