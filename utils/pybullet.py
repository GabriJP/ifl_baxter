from collections.abc import Iterator
from contextlib import contextmanager

import pybullet as p


@contextmanager
def pybullet_context(delta_t: float = 100.0) -> Iterator[None]:
    p.connect(p.DIRECT)
    p.setTimeStep(1.0 / delta_t)
    try:
        yield
    finally:
        p.resetSimulation()
        p.disconnect()
