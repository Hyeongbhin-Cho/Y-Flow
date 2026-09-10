"""POV endpoint-projection proof-of-concept experiment.

The projection implementation depends on the robot-only CasADi/Acados stack.
Load it lazily so geometry and sampling utilities remain importable in lightweight
environments used for CPU-only integrity tests.
"""

from typing import Any


__all__ = ["ProjectionDiagnostics", "TrajectoryProjector"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .projection import ProjectionDiagnostics, TrajectoryProjector

        return {
            "ProjectionDiagnostics": ProjectionDiagnostics,
            "TrajectoryProjector": TrajectoryProjector,
        }[name]
    raise AttributeError(name)
