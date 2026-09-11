"""Durable checkpoint/resume for the bonsai training pipeline.

Goal: a Colab session death costs at most ~5 minutes of work. Any run can
call ``CheckpointManager.load_latest()`` at startup and continue exactly
where the last successful ``save(state, step)`` left off.

Atomicity protocol (same shape whether the store is the HF Hub or a local
directory -- the invariant is *pointer published last*):

  1. Write every artifact into a fresh temp dir (never touch the store).
  2. sha256-checksum each file as it is written; embed the map in a
     ``manifest.json`` (the completeness marker).
  3. Publish the whole checkpoint as an immutable ``checkpoint-<step>/``
     directory.
  4. Only then publish the mutable ``latest`` pointer file -> makes the new
     checkpoint visible atomically to readers.
  5. Verify remote/store readability by listing; finally apply retention
     (keep the N=2 most recent COMPLETE checkpoints).

A crash anywhere before step 4 leaves an orphaned/incomplete directory
without a pointer bump -- ``load_latest()`` follows the pointer and verifies
integrity, so the incomplete checkpoint is ignored (and cleaned up later).

Contents saved: adapter/model weights, optimizer state_dict, LR-scheduler
state, step counter, tokens consumed, RNG states (torch/numpy/random),
data cursor.

Storage backends:
  * HF Hub private repo   (production; ``BONSAI_CKPT_REPO``, default
    ``maxta85/bonsai-checkpoints``, created if missing).
  * Local directory       (fallback + tests; ``BONSAI_CKPT_LOCAL=1`` or
    ``BONSAI_CKPT_DIR``, default ``/content/ckpt`` when it exists,
    otherwise ``./ckpt``). Uses the identical protocol: complete dir ->
    ``latest`` pointer last.

If the Hub is unreachable or unauthenticated at save time we degrade to the
local backend with a loud warning instead of dying.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any

import torch

DEFAULT_REPO_ID = "maxta85/bonsai-checkpoints"
LOCAL_FLAG_ENV = "BONSAI_CKPT_LOCAL"      # =1 -> force local backend
REPO_ENV = "BONSAI_CKPT_REPO"             # HF repo id
DIR_ENV = "BONSAI_CKPT_DIR"               # local checkpoint root dir

# Files making up one checkpoint directory.
WEIGHTS_FILE = "weights.pt"          # serialized with torch.save
OPTIMIZER_FILE = "optimizer.pt"
SCHEDULER_FILE = "scheduler.pt"
RNG_FILE = "rng.pt"
STATE_FILE = "training_state.json"
MANIFEST_FILE = "manifest.json"
POINTER_NAME = "latest"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _capture_rng_states() -> dict[str, Any]:
    """Snapshot torch / numpy / stdlib-random RNG state."""
    states: dict[str, Any] = {
        "torch": _serialize_tensor(torch.get_rng_state()),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = [
            _serialize_tensor(t)          # per-device uint8 byte strings
            for t in torch.cuda.get_rng_state_all()]
    try:
        import numpy as np
        states["numpy"] = np.random.get_state()
    except ImportError:  # pragma: no cover - numpy ships with transformers
        states["numpy"] = None
    states["python"] = random.getstate()
    return states


def _restore_rng_states(states: dict[str, Any]) -> None:
    if not states:
        return
    t = states.get("torch")
    if t is not None:
        torch.set_rng_state(_deserialize_tensor(t))
    tc = states.get("torch_cuda")
    if tc is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(
                [_deserialize_tensor(row) for row in tc])
        except Exception:  # device topology changed between sessions
            pass
    npy = states.get("numpy")
    if npy is not None:
        try:
            import numpy as np
            np.random.set_state(npy)   # MT19937 tuple round-trips via JSON
        except Exception:
            pass
    py = states.get("python")
    if py is not None:
        random.setstate(py)


def _to_jsonable(obj: Any) -> Any:
    """JSON-safe containers recursively; tensors become plain lists."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, torch.Tensor):
        return _to_jsonable(obj.detach().cpu().tolist())
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return _to_jsonable(obj.item())  # 0-d tensor / scalar types
        except Exception:
            return repr(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return repr(obj)


# --- serialization helpers --------------------------------------------------

def _serialize_tensor(tensor: torch.Tensor) -> list[int]:
    return tensor.cpu().flatten().tolist()


def _deserialize_tensor(data: list[int]) -> torch.Tensor:
    return torch.tensor(data, dtype=torch.uint8)


class CheckpointManager:
    """Save/load resumable training state with an atomic pointer protocol.

    Args:
        repo_id: HF Hub repo id (defaults to ``$BONSAI_CKPT_REPO`` or
            ``maxta85/bonsai-checkpoints``). Ignored in local mode.
        local_dir: explicit local checkpoint root (implies local mode).
        keep: how many newest COMPLETE checkpoints to retain.
        token: optional HF token (else standard resolution applies).
    """

    def __init__(
        self,
        repo_id: str | None = None,
        *,
        local_dir: str | os.PathLike | None = None,
        keep: int = 2,
        token: str | None = None,
    ):
        self.repo_id = repo_id or os.environ.get(REPO_ENV, DEFAULT_REPO_ID)
        self.token = token
        self.keep = max(1, int(keep))

        forced_local = (
            local_dir is not None
            or os.environ.get(LOCAL_FLAG_ENV, "0") == "1"
        )
        self.local_dir = Path(local_dir or os.environ.get(DIR_ENV, "") or
                              self._default_local_root())
        self.use_hub = not forced_local

    @staticmethod
    def _default_local_root() -> str:
        content = Path("/content")
        if content.is_dir():          # Colab
            return str(content / "ckpt")
        return str(Path.cwd() / "ckpt")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, state: dict[str, Any], step: int) -> dict[str, Any]:
        """Atomically persist ``state`` for ``step``.

        Returns ``{"backend": "hub"|"local", "step": step}``. Raises only
        after the full payload is durable; a failed save never advances the
        pointer so the previous checkpoint stays authoritative.
        """
        if self.use_hub:
            try:
                return self._save_hub(state, step)
            except Exception as exc:  # noqa: BLE001 - degrade, don't die
                warnings.warn(
                    f"[checkpoint] HF Hub save failed ({exc!r}); "
                    f"falling back to local-only saves under {self.local_dir}",
                    RuntimeWarning,
                )
                self.use_hub = False
        return self._save_local(state, step)

    def load_latest(self) -> dict[str, Any] | None:
        """Restore the newest COMPLETE checkpoint.

        Returns the restored state dict, updated in-place with the resolved
        ``step`` and ``cursor`` keys, or ``None`` when nothing restorable
        exists. Incomplete/corrupt candidates are skipped (crash safety).
        """
        if self.use_hub:
            try:
                result = self._load_latest_hub()
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"[checkpoint] HF Hub read failed ({exc!r}); trying "
                    f"local store {self.local_dir}", RuntimeWarning)
                result = self._load_latest_local()
            if result is not None:
                return result
            # Nothing restorable on the Hub -> still check any earlier
            # local fallback saves made by a degraded session.
            return self._load_latest_local()
        return self._load_latest_local()

    def restore_from(self, ckpt_path: str | os.PathLike) -> dict[str, Any]:
        """Read a checkpoint directory (already materialized locally).

        Verifies checksums from manifest.json. Used both by the Hub loader
        (after snapshot_download) and directly for local stores.
        """
        path = Path(ckpt_path)
        manifest_p = path / MANIFEST_FILE
        manifest = json.loads(manifest_p.read_text())
        files = manifest.get("files", {})
        for name, expected_sha in files.items():
            f = path / name
            assert f.exists(), f"checkpoint file missing: {name}"
            got = _sha256(f)
            assert got == expected_sha, (
                f"checksum mismatch for {name}: {got} != {expected_sha}")
        state = {
            "weights": {} if not (path / WEIGHTS_FILE).exists() else
            torch.load(path / WEIGHTS_FILE, map_location="cpu",
                       weights_only=False),
            "optimizer": {} if not (path / OPTIMIZER_FILE).exists() else
            torch.load(path / OPTIMIZER_FILE, map_location="cpu",
                       weights_only=False),
            "scheduler": {} if not (path / SCHEDULER_FILE).exists() else
            torch.load(path / SCHEDULER_FILE, map_location="cpu",
                       weights_only=False),
            "rng": {} if not (path / RNG_FILE).exists() else
            torch.load(path / RNG_FILE, map_location="cpu",
                       weights_only=False),
            **json.loads((path / STATE_FILE).read_text()),
        }
        return state

    def apply(self, state: dict[str, Any], *, model=None, optimizer=None,
              scheduler=None, restore_rng: bool = True,
              dataset_hash: str | None = None) -> tuple[int, Any]:
        """Convenience wrapper: push a loaded checkpoint into live objects.

        Restores RNG states by default; returns ``(step, cursor)``.

        If ``dataset_hash`` is provided and the checkpoint contains a
        ``dataset_hash`` field, they must match or a ``ValueError`` is
        raised — this prevents resuming with a different dataset.
        """
        # Validate dataset fingerprint before touching any state.
        ckpt_hash = state.get("dataset_hash")
        if dataset_hash is not None and ckpt_hash is not None:
            if dataset_hash != ckpt_hash:
                raise ValueError(
                    f"dataset hash mismatch: checkpoint has {ckpt_hash}, "
                    f"but caller expected {dataset_hash}. Refusing to "
                    f"resume with a different dataset.")
        if model is not None and state.get("weights"):
            missing, unexpected = model.load_state_dict(
                state["weights"], strict=False)
            assert not unexpected, f"unexpected checkpoint keys: {unexpected}"
        if optimizer is not None and state.get("optimizer"):
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        if restore_rng and state.get("rng"):
            _restore_rng_states(state["rng"])
        return int(state["step"]), state.get("cursor")

    # ------------------------------------------------------------------
    # Writing (shared staging logic)
    # ------------------------------------------------------------------

    def _stage_checkpoint(self, state: dict[str, Any], step: int,
                          tmpdir: Path) -> dict[str, str]:
        """Write all artifacts into ``tmpdir`` + checksums. Returns sha map."""
        state_json: dict[str, Any] = {
            "step": int(step),
            "tokens_consumed": _to_jsonable(state.get("tokens_consumed")),
            "cursor": _to_jsonable(state.get("cursor")),
        }
        if state.get("dataset_hash") is not None:
            state_json["dataset_hash"] = state["dataset_hash"]
        (tmpdir / STATE_FILE).write_text(json.dumps(state_json, indent=2))

        checksums: dict[str, str] = {}

        def write_blob(name: str, payload: Any) -> None:
            p = tmpdir / name
            torch.save(payload, p)
            checksums[name] = _sha256(p)

        def write_json(name: str, payload: Any) -> None:
            p = tmpdir / name
            p.write_text(json.dumps(payload, indent=2))
            checksums[name] = _sha256(p)

        weights = state.get("weights", {})
        if weights:
            write_blob(WEIGHTS_FILE, weights)
        else:
            raise ValueError("checkpoint state requires 'weights'")
        write_blob(OPTIMIZER_FILE, state.get("optimizer"))
        write_blob(SCHEDULER_FILE, state.get("scheduler"))
        write_blob(RNG_FILE, state.get("rng", _capture_rng_states()))
        checksums[STATE_FILE] = _sha256(tmpdir / STATE_FILE)

        write_json(MANIFEST_FILE, {"complete": True, "step": step,
                                   "files": checksums})
        return checksums

    # ------------------------------------------------------------------
    # Local backend
    # ------------------------------------------------------------------

    def _save_local(self, state: dict[str, Any], step: int) -> dict[str, Any]:
        self.local_dir.mkdir(parents=True, exist_ok=True)
        final_dir = self.local_dir / f"checkpoint-{int(step)}"
        # Stage OUTSIDE the store; move in only once complete...
        stage = Path(tempfile.mkdtemp(prefix=f".stage-ckpt-{int(step)}-",
                                      dir=str(self.local_dir)))
        moved_pointer = False
        try:
            self._stage_checkpoint(state, step, stage)
            if final_dir.exists():   # immutable per step: replace fully
                shutil.rmtree(final_dir)
            shutil.move(str(stage), str(final_dir))  # complete dir lands
            # ...pointer published LAST makes it visible atomically.
            ptr_tmp = self.local_dir / f".latest.tmp-{os.getpid()}-{step}"
            ptr_tmp.write_text(json.dumps({"step": int(step),
                                           "dir": final_dir.name}))
            os.replace(ptr_tmp, self.local_dir / POINTER_NAME)
            moved_pointer = True
            # Verify readability by listing.
            listed = sorted(p.name for p in self.local_dir.iterdir())
            assert POINTER_NAME in listed and final_dir.name in listed, \
                "local verification failed"
            self._gc_local()
            return {"backend": "local", "step": int(step)}
        finally:
            if not moved_pointer:
                shutil.rmtree(stage, ignore_errors=True)
            if not moved_pointer:
                # clean orphan pointer temps on failure paths
                for stray in self.local_dir.glob(".latest.tmp-*"):
                    stray.unlink(missing_ok=True)

    def _iter_complete_dirs(self, root: Path) -> list[tuple[int, Path]]:
        out: list[tuple[int, Path]] = []
        if not root.is_dir():
            return out
        for d in root.iterdir():
            if not (d.is_dir() and d.name.startswith("checkpoint-")):
                continue
            m = d / MANIFEST_FILE
            if not m.exists():
                continue                      # incomplete (no manifest)
            try:
                mf = json.loads(m.read_text())
                if mf.get("complete") is not True:
                    continue
                step = int(d.name.split("-", 1)[1])
                out.append((step, d))
            except Exception:
                continue                      # corrupt manifest -> skip
        out.sort(key=lambda t: t[0])
        return out

    def _load_latest_local(self) -> dict[str, Any] | None:
        root = self.local_dir
        ptr = root / POINTER_NAME
        if ptr.exists():
            try:
                name = json.loads(ptr.read_text())["dir"]
                cand = root / name
                if (cand / MANIFEST_FILE).exists():
                    state = self.restore_from(cand)
                    self._gc_local()
                    return state
            except Exception as exc:  # noqa: BLE001 - pointer stale/broken
                warnings.warn(
                    f"[checkpoint] pointer unreadable ({exc}); scanning dirs",
                    RuntimeWarning)
        # Pointer missing/unusable: fall back to newest dir that VERIFIES.
        cands = self._iter_complete_dirs(root)
        for _, cand in reversed(cands):
            try:
                state = self.restore_from(cand)
            except Exception as exc:  # noqa: BLE001 - poisoned/torn dir
                warnings.warn(
                    f"[checkpoint] skipping corrupt {cand.name} ({exc}); "
                    "falling back to older checkpoint", RuntimeWarning)
                continue
            self._gc_local()
            return state
        return None

    def _gc_local(self) -> None:
        """Keep newest N complete checkpoints; drop everything older."""
        cands = self._iter_complete_dirs(self.local_dir)
        for _, old in cands[:-self.keep] if len(cands) > self.keep else []:
            shutil.rmtree(old, ignore_errors=True)
        # Sweep incomplete leftovers (dirs without a valid manifest).
        if self.local_dir.is_dir():
            keep_names = {d.name for _, d in cands}
            for d in self.local_dir.iterdir():
                if (d.is_dir() and d.name.startswith("checkpoint-")
                        and d.name not in keep_names):
                    shutil.rmtree(d, ignore_errors=True)  # never-published
                elif d.name.startswith(".stage-ckpt-"):
                    shutil.rmtree(d, ignore_errors=True)

    # ------------------------------------------------------------------
    # Hub backend
    # ------------------------------------------------------------------

    def _hub_api(self):
        from huggingface_hub import HfApi
        return HfApi(token=self.token)

    def _save_hub(self, state: dict[str, Any], step: int) -> dict[str, Any]:
        from huggingface_hub import (HfApi, create_repo, upload_folder,
                                     upload_file, list_repo_files)

        api: HfApi = self._hub_api()
        repo_id = self.repo_id
        private = True
        created = False
        try:
            api.create_repo(repo_id, private=private, exist_ok=True)
            created = True
        except Exception as exc:  # noqa: BLE001 - unauthed/no-quota etc.
            if "409" in repr(exc) or "You already created" in str(exc):
                created = True          # races are fine with exist_ok
            else:
                raise

        step_s = str(int(step))
        remote_dir = f"checkpoint-{step_s}"

        with tempfile.TemporaryDirectory(
                prefix="bonsai-ckpt-hub-") as td:
            stage = Path(td) / remote_dir
            stage.mkdir(parents=True)
            self._stage_checkpoint(state, step, stage)

            if not created:
                # ensure repo exists even when probe above skipped us
                api.create_repo(repo_id, private=private, exist_ok=True)
            # Immutable checkpoint dir upload.
            upload_folder(
                folder_path=str(stage),
                repo_id=repo_id,
                path_in_repo=remote_dir,
                commit_message=f"bonsai ckpt step {step}",
                token=self.token,
            )
            # Pointer publish LAST (atomic visibility of new checkpoint).
            ptr_stage = Path(td) / "_ptr"
            ptr_stage.mkdir()
            (ptr_stage / POINTER_NAME).write_text(
                json.dumps({"step": int(step), "dir": remote_dir}))
            upload_file(
                path_or_fileobj=str(ptr_stage / POINTER_NAME),
                path_in_repo=POINTER_NAME,
                repo_id=repo_id,
                commit_message=f"bonsai latest pointer -> {remote_dir}",
                token=self.token,
            )

        # Remote readability verification by listing.
        files = list_repo_files(repo_id, token=self.token)
        assert f"{remote_dir}/{MANIFEST_FILE}" in files, \
            f"upload verification failed: no {MANIFEST_FILE} remotely"
        assert POINTER_NAME in files, "pointer missing after publish"

        self._gc_hub(api, repo_id)
        return {"backend": "hub", "step": int(step)}

    def _hub_download(self, api, repo_id: str, remote_dir: str,
                      target: Path) -> Path | None:
        from huggingface_hub import snapshot_download
        allow = [
            f"{remote_dir}/{n}" for n in
            (WEIGHTS_FILE, OPTIMIZER_FILE, SCHEDULER_FILE, RNG_FILE,
             STATE_FILE, MANIFEST_FILE)
        ]
        try:
            got = snapshot_download(
                repo_id=repo_id, allow_patterns=allow, token=self.token,
                local_dir=target)
        except Exception:
            return None
        dl = Path(got) / remote_dir
        return dl if (dl / MANIFEST_FILE).exists() else None

    def _list_remote_dirs(self, api, repo_id: str) -> dict[str, set[str]]:
        """Map remote 'checkpoint-<n>' -> set of its files."""
        from huggingface_hub import list_repo_files
        try:
            files = list_repo_files(repo_id, token=self.token)
        except Exception:
            return {}
        dirs: dict[str, set[str]] = {}
        for f in files:
            parts = f.split("/")
            if len(parts) >= 2 and parts[0].startswith("checkpoint-"):
                dirs.setdefault(parts[0], set()).update(parts[1:])
        return dirs

    def _load_latest_hub(self) -> dict[str, Any] | None:
        api = self._hub_api()
        repo_id = self.repo_id
        dirs = self._list_remote_dirs(api, repo_id)
        if not dirs:
            return None

        pointer = None
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(repo_id, POINTER_NAME, token=self.token)
            pointer = json.loads(Path(p).read_text())
        except Exception:
            pointer = None  # crash before first publish -> scan instead

        candidates: list[int] = []
        for name in dirs:
            if MANIFEST_FILE in dirs[name]:      # complete marker present
                try:
                    candidates.append(int(name.split("-", 1)[1]))
                except ValueError:
                    pass
        if not candidates:
            return None
        candidates.sort()

        order = [pointer["step"]] if pointer and pointer["step"] in candidates else []
        order += [s for s in reversed(candidates) if s not in order]

        tmp_root = None
        try:
            for step in order:
                remote_dir = f"checkpoint-{step}"
                if MANIFEST_FILE not in dirs.get(remote_dir, ()):
                    continue                     # incomplete -> skip
                tmp_root = Path(tempfile.mkdtemp(prefix="bonsai-ckpt-dl-"))
                dl = self._hub_download(api, repo_id, remote_dir, tmp_root)
                if dl is None:
                    continue
                try:
                    state = self.restore_from(dl)   # verifies checksums too
                    return state
                except AssertionError:
                    continue                        # corrupt -> next
                finally:
                    shutil.rmtree(tmp_root, ignore_errors=True)
                    tmp_root = None
        finally:
            if tmp_root is not None:
                shutil.rmtree(tmp_root, ignore_errors=True)
        return None

    def _gc_hub(self, api, repo_id: str) -> None:
        """Keep newest N complete checkpoint dirs on the Hub; delete rest."""
        dirs = self._list_remote_dirs(api, repo_id)
        complete = []
        for name in sorted(dirs):
            try:
                step = int(name.split("-", 1)[1])
            except ValueError:
                step = -1
            if step >= 0 and MANIFEST_FILE in dirs[name]:
                complete.append((step, name))
        complete.sort()
        for _, name in complete[:-self.keep] if len(complete) > self.keep else []:
            try:
                for fname in sorted(dirs[name]):
                    api.delete_file(f"{name}/{fname}", repo_id=repo_id,
                                    token=self.token)
            except Exception:
                pass  # retention is best-effort
        # Also remove stale pointer-free incomplete dirs left by crashes.
        keep = {n for _, n in complete}
        for name in sorted(set(dirs) - keep):
            if POINTER_NAME == name:
                continue
            if (dirs[name] and MANIFEST_FILE not in dirs[name]) or name.endswith("_wip"):
                try:
                    for fname in sorted(dirs[name]):
                        api.delete_file(f"{name}/{fname}", repo_id=repo_id,
                                        token=self.token)
                except Exception:
                    pass


def build_state(model=None, *, weights: dict | None = None,
                optimizer=None, scheduler=None, tokens_consumed: int = 0,
                cursor: Any = None, dataset_hash: str | None = None) -> dict[str, Any]:
    """Assemble a save-ready ``state`` payload from live training objects.

    Weight tensors are CLONED (snapshot semantics): the returned payload no
    longer tracks live parameters, so later optimizer steps cannot mutate
    an already-built state.

    ``weights`` wins over ``model.state_dict()`` when both given (e.g. a
    pre-extracted LoRA adapter sub-dict).

    ``dataset_hash`` is an optional fingerprint of the training dataset
    (e.g. sha256 of the tokenized data). When present in a checkpoint,
    ``apply()`` validates that the expected hash matches the one provided
    by the caller, rejecting mismatches to prevent silent data changes
    across resume sessions.
    """
    if model is not None and weights is None:
        weights = {k: v.detach().cpu().clone()
                   for k, v in model.state_dict().items()}
    elif weights is not None:
        weights = {k: (v.detach().cpu().clone()
                       if isinstance(v, torch.Tensor) else v)
                   for k, v in weights.items()}
    # optimizer.state_dict() returns REFERENCES to live moment tensors — clone
    # them (snapshot semantics) or later optimizer steps mutate the payload.
    opt_state = optimizer.state_dict() if optimizer is not None else {}
    if opt_state:
        opt_state = {
            "state": {pid: {k: (v.detach().cpu().clone()
                                if isinstance(v, torch.Tensor) else v)
                             for k, v in st.items()}
                       for pid, st in opt_state.get("state", {}).items()},
            "param_groups": [dict(g) for g in opt_state.get("param_groups", [])],
        }
    state: dict[str, Any] = {
        "weights": weights or {},
        "optimizer": opt_state,
        "scheduler": scheduler.state_dict() if scheduler is not None else {},
        "tokens_consumed": tokens_consumed,
        "cursor": cursor,
        "rng": _capture_rng_states(),
    }
    if dataset_hash is not None:
        state["dataset_hash"] = dataset_hash
    return state


    def apply(self, state: dict[str, Any], *, model=None, optimizer=None,
              scheduler=None, restore_rng: bool = True) -> tuple[int, Any]:
        """Restore a loaded checkpoint payload onto live training objects.

        Returns (step, cursor). Uses in-place tensor copies so objects the
        caller already holds (optimizer parameter groups etc.) stay valid.
        """
        if state is None:
            raise ValueError("apply() called with None state")

        step = int(state.get("step", 0))
        cursor = state.get("cursor")

        weights = state.get("weights") or {}
        if model is not None and weights:
            sd = model.state_dict()
            for k, v in weights.items():
                if k in sd and isinstance(v, torch.Tensor):
                    sd[k].copy_(v.to(sd[k].device, sd[k].dtype))
            model.load_state_dict(sd, strict=False)

        opt_state = state.get("optimizer") or {}
        if optimizer is not None and opt_state:
            optimizer.load_state_dict(opt_state)

        sched_state = state.get("scheduler") or {}
        if scheduler is not None and sched_state:
            scheduler.load_state_dict(sched_state)

        if restore_rng:
            rng = state.get("rng") or {}
            _restore_rng_states(rng)

        return step, cursor

    def _resolve_state_file(self, state: dict[str, Any] | None) -> Path | None:
        """Compatibility helper: locate the manifest path inside a loaded payload."""
        if state is None:
            return None
        mf = state.get("manifest_path")
        return Path(mf) if mf else None


__all__ = [
    "CheckpointManager", "build_state",
    "DEFAULT_REPO_ID", "POINTER_NAME", "MANIFEST_FILE",
]
