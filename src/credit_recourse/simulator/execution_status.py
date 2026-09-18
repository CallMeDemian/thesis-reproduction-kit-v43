from __future__ import annotations


class SimulatorExecutionRejected(ValueError):
    """A documented economic feasibility rejection with a stable status code."""

    def __init__(self, status_code: str, detail: str):
        self.status_code = str(status_code)
        self.detail = str(detail)
        super().__init__(f"{self.status_code}: {self.detail}")


def accepted_rejection_code(exc: BaseException, allowed_codes: set[str] | tuple[str, ...]) -> str | None:
    if not isinstance(exc, SimulatorExecutionRejected):
        return None
    return exc.status_code if exc.status_code in set(allowed_codes) else None
