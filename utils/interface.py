import logging
from pathlib import Path
from typing import Final

import click

logger: Final = logging.getLogger("cen_brnn")


@click.group()
@click.option("-v", "--verbose", is_flag=True)
@click.option("--log-to-txt", is_flag=True)
def cli(verbose: bool, log_to_txt: bool) -> None:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler = logging.StreamHandler()

    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    txt_logger = logging.getLogger("txt_logger")
    txt_logger.setLevel(logging.INFO)
    txt_logger.propagate = False

    if log_to_txt:
        txt_logger.propagate = True
        append_handler = logging.FileHandler("txt_logger.log")
        append_handler.setFormatter(formatter)
        txt_logger.addHandler(append_handler)


def get_wandb_config_data(
    train_paths: tuple[Path],
    test_paths: tuple[Path],
    epochs: int | None,
    batch_size: int | None,
) -> dict[str, int | float | str | None]:
    config_dict: dict[str, int | float | str | None] = dict(
        epochs=epochs,
        batch_size=batch_size,
    )
    config_dict.update({f"train_path_{i}": str(tp) for i, tp in enumerate(train_paths)})
    config_dict.update({f"test_path_{i}": str(tp) for i, tp in enumerate(test_paths)})
    return config_dict
