"""Deep Q-Network agent for the Gymnasium ``LunarLander-v3`` environment.

This package contains all of the project's real logic. The modules in
``scripts/`` are thin command-line wrappers around it.

Module map:

* :mod:`lunar_lander_ai.config`          - YAML + CLI to validated frozen dataclasses.
* :mod:`lunar_lander_ai.environment`     - ``make_env`` and runtime space verification.
* :mod:`lunar_lander_ai.network`         - the :class:`QNetwork` function approximator.
* :mod:`lunar_lander_ai.replay_buffer`   - preallocated experience-replay ring buffer.
* :mod:`lunar_lander_ai.agent`           - the :class:`DQNAgent`.
* :mod:`lunar_lander_ai.train`           - the episode loop and checkpoint scheduling.
* :mod:`lunar_lander_ai.evaluate`        - greedy, weight-frozen evaluation.
* :mod:`lunar_lander_ai.rewards`         - optional reward-shaping wrapper.
* :mod:`lunar_lander_ai.metrics`         - landing detection and summary statistics.
* :mod:`lunar_lander_ai.plotting`        - training curves from real logs.
* :mod:`lunar_lander_ai.checkpoint`      - atomic checkpoint save/load with provenance.
* :mod:`lunar_lander_ai.utils`           - seeding, device selection, paths.

The implementation contract is ``docs/BUILD_SPEC.md``.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.2.1"
