import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import F64_A
from . import Trajectory


def load_trajectory(trajectory_paths: Sequence[Path]) -> tuple[Trajectory, ...]:
    descriptors: list[F64_A] = []
    paths: list[F64_A] = []

    for trajectory_path in trajectory_paths:
        paths.append(np.loadtxt(trajectory_path, delimiter=" ", skiprows=1)[::5])
        with trajectory_path.with_suffix(".npy").open("rb") as f:
            descriptors.append(pickle.load(f))

    return tuple(
        Trajectory(t_p.stem, descriptor, path)
        for t_p, descriptor, path in zip(trajectory_paths, descriptors, paths, strict=False)
    )


def load_gen_data(
    train_paths: Sequence[Path], test_paths: Sequence[Path]
) -> tuple[tuple[Trajectory, ...], tuple[Trajectory, ...]]:
    if not len(train_paths) or not len(test_paths):
        raise ValueError

    train_data = load_trajectory(train_paths)
    test_data = load_trajectory(test_paths)

    #  Generate only if any of the paths is not given

    if train_data is None or test_data is None:
        raise ValueError

    return train_data, test_data
