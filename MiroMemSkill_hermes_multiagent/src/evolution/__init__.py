# SPDX-FileCopyrightText: 2026 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Offline skill-evolution control plane (Hermes-inspired, rollout-driven).

Design contract:
- The runtime (orchestrator / MCP tools / evaluators) is never imported from
  here at module load time; arms run as subprocesses of ``main.py``.
- Candidates are immutable content-addressed artifacts; the production skill
  file is only ever changed through ``SkillRegistry.promote`` (CAS-guarded).
- Fitness comes from the deterministic financial replay, never from an LLM.
"""
