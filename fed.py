import logging
import time
from collections.abc import Callable
from logging import INFO
from pathlib import Path
from statistics import mean

import click
import flwr as fl
import wandb
from flwr.common import Metrics
from flwr.common import NDArrays
from flwr.common import Scalar
from flwr.common.logger import log
from flwr.server.strategy import FedAvg

from models import BrnnModel
from utils import cli
from utils import get_wandb_config_data
from utils import load_gen_data
from utils import save_model_wandb
from utils.fed import BrnnClient
from utils.fed import CubeStrategy
from utils.fed import LightParallelClient


@cli.command()
@click.argument("server_address", type=str, default="localhost:8080")
@click.option("--wandb-project", default="baxter")
@click.option("--wandb-group", required=True)
@click.option("--wandb-name", required=True)
@click.option("--train-paths", type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path))
@click.option(
    "--test-paths", type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path), multiple=True
)
@click.option("--light", is_flag=True)
def client(
    server_address: str,
    wandb_project: str,
    wandb_group: str,
    wandb_name: str,
    train_paths: Path,
    test_paths: tuple[Path],
    light: bool,
) -> None:
    time.sleep(5)

    wandb.init(
        project=wandb_project,
        entity="gabijp",
        group=wandb_group,
        name=wandb_name,
        config=get_wandb_config_data((train_paths,), test_paths, None, None),
    )

    train_data, test_data = load_gen_data((train_paths,), test_paths)

    brnn_inv = BrnnModel(tx=25)
    brnn_inv.compile()
    brnn_inv.build()

    fl_client = (
        LightParallelClient(brnn_inv, train_data, test_data) if light else BrnnClient(brnn_inv, train_data, test_data)
    )

    fl.client.start_client(
        server_address=server_address,
        client=fl_client.to_client(),
        root_certificates=(Path.home() / "certs" / "ca.crt").read_bytes(),
    )


def on_fit_config_generator(default_config: dict[str, Scalar]) -> Callable[[int], dict[str, Scalar]]:
    def inner(round_n: int) -> dict[str, Scalar]:
        res = default_config.copy()
        res.update(
            dict(
                epochs=(round_n + 1) * default_config["epochs"],
                current_epoch=round_n * default_config["epochs"],
            )
        )
        return res

    return inner


def on_evaluate_config_generator(n_rounds: int) -> Callable[[int], dict[str, Scalar]]:
    def inner(round_n: int) -> dict[str, Scalar]:
        # round_n starts at 1, so last one equals n_rounds
        do_eval = round_n >= n_rounds
        if do_eval:
            log(INFO, "on_evaluate_config_generator: sending do_eval positive message")
        return dict(do_eval=do_eval)

    return inner


def savewb_evaluate_function_generator(total_server_rounds: int) -> Callable[[int, NDArrays, dict[str, Scalar]], None]:
    brnn_inv = BrnnModel(tx=25)
    brnn_inv.compile()
    brnn_inv.build()

    # server_round starts at 1, so last one equals total_server_rounds
    def inner(server_round: int, parameters: NDArrays, _: dict[str, Scalar]) -> None:
        msg = f"round {server_round}/{total_server_rounds}"
        logging.info(msg)
        if total_server_rounds != server_round:
            return
        brnn_inv.set_weights(parameters)
        save_model_wandb(brnn_inv)

    return inner


def average_metrics_generator(log_wandb: bool) -> Callable[[list[tuple[int, Metrics]]], Metrics]:
    def average_metrics(metrics: list[tuple[int, Metrics]]) -> Metrics:
        mae_keys = ("mae_s0", "mae_s1", "mae_e0", "mae_e1", "mae_w0", "mae_w1", "mae_w2", "mae_mean", "r2")

        aggregated_metrics = dict()
        for key in mae_keys:
            aggregated_metrics[key] = mean(m[key] for _, m in metrics)

        if log_wandb and any(m.get("do_eval", False) for _, m in metrics):
            wandb.log(dict(agg=aggregated_metrics))

        return aggregated_metrics

    return average_metrics


@cli.command(context_settings=dict(show_default=True))
@click.option("--port", type=click.IntRange(1, 65_535), default=8080)
@click.option("--wandb-project", default="baxter")
@click.option("--wandb-group", required=True)
@click.option("--num-rounds", type=click.IntRange(1), default=5)
@click.option("--epochs", type=click.IntRange(1), default=2)
@click.option("--batch-size", type=click.IntRange(1), default=32)
@click.option("--min-fit-clients", type=click.IntRange(1), default=2)
@click.option("--min-evaluate-clients", type=click.IntRange(0), default=2)
@click.option("--min-available-clients", type=click.IntRange(1), default=2)
@click.option("--cube", is_flag=True)
@click.option("--quantity", is_flag=True)
def server(
    port: int,
    wandb_project: str,
    wandb_group: str,
    num_rounds: int,
    epochs: int,
    batch_size: int,
    min_fit_clients: int,
    min_evaluate_clients: int,
    min_available_clients: int,
    cube: bool,
    quantity: bool,
) -> None:
    wandb_agg_mode = "fedavg"
    if cube:
        wandb_agg_mode = f"cube_{'q' if quantity else 'noq'}"

    wandb.init(
        project=wandb_project,
        entity="gabijp",
        group=wandb_group,
        name="server",
        config=dict(
            num_rounds=num_rounds,
            epochs=epochs,
            batch_size=batch_size,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            agg=wandb_agg_mode,
        ),
    )

    if cube:
        strategy = CubeStrategy(
            fraction_fit=0.0,
            fraction_evaluate=0.0 if min_evaluate_clients == 0 else 1e-5,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=savewb_evaluate_function_generator(num_rounds),
            on_fit_config_fn=on_fit_config_generator(dict(epochs=epochs, current_epoch=0, batch_size=batch_size)),
            on_evaluate_config_fn=on_evaluate_config_generator(num_rounds),
            accept_failures=False,
            fit_metrics_aggregation_fn=lambda _: dict(),
            evaluate_metrics_aggregation_fn=average_metrics_generator(True),
            q=quantity,
        )
    else:
        strategy = FedAvg(
            fraction_fit=0.0,
            fraction_evaluate=0.0 if min_evaluate_clients == 0 else 1e-5,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=savewb_evaluate_function_generator(num_rounds),
            on_fit_config_fn=on_fit_config_generator(dict(epochs=epochs, current_epoch=0, batch_size=batch_size)),
            on_evaluate_config_fn=on_evaluate_config_generator(num_rounds),
            accept_failures=False,
            fit_metrics_aggregation_fn=lambda _: dict(),
            evaluate_metrics_aggregation_fn=average_metrics_generator(True),
        )

    certificates_path = Path.home() / "certs"
    fl.server.start_server(
        server_address=f"0.0.0.0:{port}",
        config=fl.server.ServerConfig(num_rounds=num_rounds, round_timeout=600.0),  # Timeout==10 minutes
        strategy=strategy,
        grpc_max_message_length=1024**3,  # 1 GiB
        certificates=(
            (certificates_path / "ca.crt").read_bytes(),
            (certificates_path / "server.pem").read_bytes(),
            (certificates_path / "server.key").read_bytes(),
        ),
    )


if __name__ == "__main__":
    cli()
