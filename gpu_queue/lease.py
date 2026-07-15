"""Kernel-backed interactive leases for latency-sensitive GPU work."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import time
import uuid


_SCHEMA = "gpu-greenroom.interactive-lease.v1"
_LEASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TERMINAL_STATES = {"released", "released-unacquired"}


class InteractiveLeaseError(RuntimeError):
    """Raised when an interactive lease cannot preserve its identity contract."""


class InteractiveLease:
    """Own ``gpu.lock`` while an external latency-sensitive task is active."""

    def __init__(
        self,
        queue_dir: str | Path,
        *,
        lease_id: str,
        holder: str,
        purpose: str,
        receipt_path: str | Path | None = None,
    ) -> None:
        if not _LEASE_ID_PATTERN.fullmatch(lease_id):
            raise InteractiveLeaseError(
                "lease_id must contain only letters, digits, dot, underscore, and dash"
            )
        if not holder.strip():
            raise InteractiveLeaseError("holder must not be blank")
        if not purpose.strip():
            raise InteractiveLeaseError("purpose must not be blank")

        self.queue_dir = Path(queue_dir)
        self.lock_path = self.queue_dir / "gpu.lock"
        self.lease_id = lease_id
        self.holder = holder.strip()
        self.purpose = purpose.strip()
        self.receipt_path = (
            Path(receipt_path)
            if receipt_path is not None
            else self.queue_dir / "leases" / lease_id / "receipt.json"
        )
        self._receipt: dict | None = None
        self._lock_fd = None

    @property
    def is_effective(self) -> bool:
        return self._lock_fd is not None

    def snapshot(self) -> dict:
        if self._receipt is None:
            raise InteractiveLeaseError("lease has not been requested")
        return dict(self._receipt)

    def request(self) -> dict:
        """Persist requested identity before attempting to acquire the GPU lock."""
        if self._receipt is not None:
            return self.snapshot()

        now = time.time()
        self._receipt = {
            "schema": _SCHEMA,
            "lease_id": self.lease_id,
            "holder": self.holder,
            "purpose": self.purpose,
            "state": "requested",
            "requested_at": now,
            "effective_at": None,
            "released_at": None,
            "holder_pid": os.getpid(),
            "queue_dir": str(self.queue_dir),
            "lock_path": str(self.lock_path),
            "last_trustworthy_event": "request-persisted",
            "current_authority": "requires-live-holder-process-and-flock",
        }
        try:
            self._write_initial_receipt()
        except BaseException:
            self._receipt = None
            raise
        return self.snapshot()

    def acquire(self, *, blocking: bool = True) -> bool:
        """Claim ``gpu.lock`` and immediately publish effectiveness."""
        if not self.claim_lock(blocking=blocking):
            return False
        self.publish_effective()
        return True

    def claim_lock(self, *, blocking: bool = True) -> bool:
        """Claim ``gpu.lock`` without publishing an effective lease yet."""
        if self._receipt is None:
            self.request()
        if self._lock_fd is not None:
            return True
        assert self._receipt is not None
        if self._receipt["state"] in _TERMINAL_STATES:
            raise InteractiveLeaseError(
                f"lease {self.lease_id} is terminal: {self._receipt['state']}"
            )

        self.queue_dir.mkdir(parents=True, exist_ok=True)
        lock_fd = open(self.lock_path, "w")
        operation = fcntl.LOCK_EX
        if not blocking:
            operation |= fcntl.LOCK_NB
        try:
            fcntl.flock(lock_fd, operation)
        except BlockingIOError:
            lock_fd.close()
            return False
        except BaseException:
            lock_fd.close()
            raise

        self._lock_fd = lock_fd
        return True

    def publish_effective(self) -> dict:
        """Persist effectiveness after the caller accepts the claimed lock."""
        if self._lock_fd is None:
            raise InteractiveLeaseError("cannot publish effective without gpu.lock")
        assert self._receipt is not None
        if self._receipt["state"] == "effective":
            return self.snapshot()
        if self._receipt["state"] != "requested":
            raise InteractiveLeaseError(
                f"cannot publish effective from state {self._receipt['state']}"
            )
        self._receipt.update(
            state="effective",
            effective_at=time.time(),
            last_trustworthy_event="gpu-lock-acquired",
        )
        self._write_receipt()
        return self.snapshot()

    def release(self) -> dict:
        """Release the kernel lock without claiming acquisition when none occurred."""
        if self._receipt is None:
            self.request()
        assert self._receipt is not None
        if self._receipt["state"] in _TERMINAL_STATES:
            return self.snapshot()

        had_lock = self._lock_fd is not None
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            finally:
                self._lock_fd.close()
                self._lock_fd = None

        was_effective = self._receipt["effective_at"] is not None
        self._receipt.update(
            state="released" if was_effective else "released-unacquired",
            released_at=time.time(),
            last_trustworthy_event=(
                "gpu-lock-released"
                if was_effective
                else (
                    "gpu-lock-released-before-effective-publication"
                    if had_lock
                    else "request-cancelled-before-acquisition"
                )
            ),
        )
        self._write_receipt()
        return self.snapshot()

    def _write_initial_receipt(self) -> None:
        """Publish the first receipt without replacing an existing lease identity."""
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._write_temp_receipt()
        try:
            os.link(temp_path, self.receipt_path)
        except FileExistsError as exc:
            raise InteractiveLeaseError(
                f"lease receipt already exists: {self.receipt_path}"
            ) from exc
        finally:
            temp_path.unlink(missing_ok=True)

    def _write_receipt(self) -> None:
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._write_temp_receipt()
        os.replace(temp_path, self.receipt_path)

    def _write_temp_receipt(self) -> Path:
        assert self._receipt is not None
        temp_path = self.receipt_path.with_name(
            f".{self.receipt_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        temp_path.write_text(json.dumps(self._receipt, indent=2) + "\n")
        return temp_path
