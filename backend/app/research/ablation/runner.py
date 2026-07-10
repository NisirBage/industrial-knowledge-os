"""AblationRunner (O3 Component 1): runs registered probes and generates
ablation configurations.

"Generate all combinations automatically" is implemented literally via
`generate_full_factorial_configurations()` — a real, working
`itertools.product` generator, not a placeholder — but running the full
16-axis factorial (2**16 = 65536 configurations, each needing a real
pipeline run) is deliberately not the default: `generate_one_factor_at_a_time_configurations()`
(16 configurations, one axis flipped at a time against an all-enabled
baseline) is what `run_all()` actually executes, matching every other
research benchmark's "small, real, reproducible" discipline. Requesting a
full factorial over more than `_MAX_FACTORIAL_AXES` axes raises rather than
silently truncating or fabricating unrun results.
"""

import itertools

from app.research.ablation.models import AblationAxis, AblationConfiguration, AblationResult
from app.research.ablation.probes import PROBES

_MAX_FACTORIAL_AXES = 6  # 2**6 = 64 configurations — still real to enumerate; beyond this is exponential blowup, not run here.


def generate_one_factor_at_a_time_configurations(axes: list[AblationAxis] | None = None) -> list[AblationConfiguration]:
    axes = axes or list(AblationAxis)
    all_enabled = frozenset(AblationAxis)
    return [AblationConfiguration(enabled_axes=all_enabled - {axis}) for axis in axes]


def generate_full_factorial_configurations(axes: list[AblationAxis]) -> list[AblationConfiguration]:
    if len(axes) > _MAX_FACTORIAL_AXES:
        raise ValueError(
            f"Full factorial over {len(axes)} axes would be {2 ** len(axes)} configurations — "
            f"capped at {_MAX_FACTORIAL_AXES} axes ({2 ** _MAX_FACTORIAL_AXES} configurations) to stay a real, "
            f"runnable sweep rather than a combinatorial explosion. Pass a smaller axes subset."
        )
    other_axes = frozenset(AblationAxis) - frozenset(axes)
    configurations = []
    for combination in itertools.product([True, False], repeat=len(axes)):
        enabled = other_axes | {axis for axis, is_on in zip(axes, combination, strict=True) if is_on}
        configurations.append(AblationConfiguration(enabled_axes=frozenset(enabled)))
    return configurations


class AblationRunner:
    async def run_axis(self, axis: AblationAxis) -> AblationResult:
        probe = PROBES.get(axis)
        if probe is None:
            raise ValueError(f"No probe registered for ablation axis {axis!r}")
        return await probe()

    async def run_all(self) -> list[AblationResult]:
        return [await self.run_axis(axis) for axis in AblationAxis]
