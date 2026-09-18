"""Canonical V4.3 final RL binding and Plan-3 LLM execution runtime.

Historical training and experiment code is intentionally excluded from this
package.  All supported public execution begins at ``python -m
credit_recourse.final_release`` and is hash-bound by ``configs/current``.
"""

from .common import ContractError, find_repo_root, sha256_file

__all__ = ["ContractError", "find_repo_root", "sha256_file"]
