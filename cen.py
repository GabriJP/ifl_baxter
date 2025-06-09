import logging
import pickle
from collections.abc import Iterable
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import wandb

from models import BrnnModel
from utils import cli
from utils import F64_A
from utils import get_path_descriptor
from utils import get_wandb_config_data
from utils import load_gen_data
from utils import save_model_wandb
from utils.fed import BrnnClient

logging.basicConfig(level=logging.INFO)


def get_xyz_ee(array_ang: F64_A, limb_name: str) -> F64_A:
    from data_generator import analytic_model as rbd

    limb_baxter = rbd.limbs[limb_name]
    xyz = []
    for ang in array_ang:
        limb_baxter.reset_joints(ang)
        xyz_i = limb_baxter.get_ee_state()
        xyz.append(xyz_i)

    return np.array(xyz)


def plot_xyz_paths(trajectories: list[F64_A], limb_name: str) -> None:
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 4)
    for traj in trajectories:
        xyz_traj = get_xyz_ee(traj[::20], limb_name)
        ax1.plot(xyz_traj[:, 0], xyz_traj[:, 1], xyz_traj[:, 2])
        ax2.plot(xyz_traj[:, 0], xyz_traj[:, 1])
        ax3.plot(xyz_traj[:, 0], xyz_traj[:, 2])
    # ax1.set_title('XYZ')
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    # ax2.set_title('XY')
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    # ax3.set_title('XZ')
    ax3.set_xlabel("X")
    ax3.set_ylabel("Z")
    plt.show()


def find_csv(trajectories_path: Path, recurrent: bool) -> Iterable[Path]:
    for path in trajectories_path.iterdir():
        if recurrent and path.is_dir():
            yield from find_csv(path, recurrent)
        elif path.is_file() and path.suffix == ".csv":
            yield path


@cli.command()
@click.argument(
    "TRAJECTORIES_PATH", type=click.Path(exists=True, file_okay=False, path_type=Path), default=Path("real_data")
)
@click.option("--recurrent", is_flag=True)
@click.option("--limb", "limb_name", type=click.Choice(["left", "right"]), required=True)
def plot(trajectories_path: Path, recurrent: bool, limb_name: str) -> None:
    dfs = [np.genfromtxt(f, delimiter=" ", skip_header=1) for f in find_csv(trajectories_path, recurrent)]

    from utils.pybullet import pybullet_context

    with pybullet_context():
        plot_xyz_paths(dfs, limb_name)


@cli.command()
@click.argument("TRAJECTORIES_PATH", type=click.Path(file_okay=False, writable=True, resolve_path=True, path_type=Path))
def describe(trajectories_path: Path) -> None:
    trajectory_paths = [p for p in trajectories_path.iterdir() if p.is_file() and p.suffix == ".csv"]
    paths = [np.loadtxt(trajectory_path, delimiter=" ", skiprows=1) for trajectory_path in trajectory_paths]

    from utils.pybullet import pybullet_context

    with pybullet_context():
        descriptors = get_path_descriptor(paths)

    for trajectory_path, descriptor in zip(trajectory_paths, descriptors, strict=False):
        with trajectory_path.with_suffix(".npy").open("wb") as f:
            pickle.dump(descriptor, f)


@cli.command()
@click.option("--wandb-project", default="baxter")
@click.option("--wandb-group", required=True)
@click.option("--wandb-name", required=True)
@click.option(
    "--train-paths", type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path), multiple=True
)
@click.option(
    "--test-paths", type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path), multiple=True
)
@click.option("--epochs", type=click.IntRange(min=1), default=1)
@click.option("--batch-size", type=click.IntRange(min=1), default=32)
def cen(
    wandb_project: str,
    wandb_group: str,
    wandb_name: str,
    train_paths: tuple[Path],
    test_paths: tuple[Path],
    epochs: int,
    batch_size: int,
) -> None:
    wandb.init(
        project=wandb_project,
        entity="gabijp",
        group=wandb_group,
        name=wandb_name,
        config=get_wandb_config_data(
            train_paths,
            test_paths,
            epochs,
            batch_size,
        ),
    )

    train_data, test_data = load_gen_data(train_paths, test_paths)

    brnn_inv = BrnnModel(tx=25)
    brnn_inv.compile()
    client = BrnnClient(brnn_inv, train_data, test_data)

    client.fit(None, dict(epochs=epochs, current_epoch=0, batch_size=batch_size))
    r2, _, mae_dict = client.evaluate()

    logging.info("R2: %s", r2)
    logging.info("MAEs: %s", mae_dict)

    save_model_wandb(brnn_inv)


if __name__ == "__main__":
    cli()
