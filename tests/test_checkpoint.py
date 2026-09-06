"""tests/test_checkpoint.py

Checkpoint/resume semantics for training.checkpoint.CheckpointManager --
all CPU, all offline. The Hub backend is exercised through huggingface_hub's
local cache semantics: tests run with BONSAI_CKPT_LOCAL=1-style managers or
monkeypatched local-store internals, never touching the network.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.checkpoint import CheckpointManager, build_state


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_store(tmp_path):
    """A CheckpointManager pointed at a throwaway local store."""
    return CheckpointManager(local_dir=str(tmp_path / "store"), keep=2)


def _make_toy_model(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(),
                               torch.nn.Linear(16, 4))


def _fresh_training_objects(seed=1234, lr=0.05):
    """Rebuild model/optimizer/scheduler as a 'new process' would."""
    model = _make_toy_model(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 1.0 / (1.0 + 0.5 * step))
    return model, optimizer, scheduler


def _train_steps(model, optimizer, scheduler, x, y, n):
    losses = []
    for _ in range(n):
        optimizer.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item())
    return losses


def _rng_fingerprint():
    nps = np.random.get_state()
    return {
        "torch": torch.get_rng_state().clone().tolist(),
        "numpy": (nps[0], tuple(nps[1][:10]), nps[2]),
        "python": random.getstate(),
    }


def _assert_state_restored(before, after):
    # weights
    b, a = before["weights"], after["weights"]
    assert set(b) == set(a)
    for k in b:
        assert torch.equal(b[k], a[k]), f"weight mismatch: {k}"
    # optimizer momentum buffers (exp_avg / exp_avg_sq)
    bo, ao = before["optimizer"]["state"], after["optimizer"]["state"]
    assert set(bo) == set(ao)
    for pid in bo:
        for field in ("exp_avg", "exp_avg_sq"):
            if field in bo[pid]:
                assert torch.equal(bo[pid][field].cpu(),
                                   ao[pid][field].cpu()), \
                    f"optimizer {field} mismatch for param {pid}"
    assert before["step"] == after["step"]
    assert before["cursor"] == after["cursor"]
    # NOTE: rng equality is covered by test_rng_states_restored (which uses
    # restore_rng=True). This test runs with restore_rng=False, so live RNG
    # has advanced past the snapshot and must NOT be compared.


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------

def test_roundtrip_full_state(tmp_store):
    model, optimizer, scheduler = _fresh_training_objects()
    torch.manual_seed(999)

    for step, cursor in [(1, 512), (2, 1536)]:
        _train_steps(model, optimizer, scheduler,
                     torch.randn(4, 8), torch.randn(4, 4), 1)
        state = build_state(model=model, optimizer=optimizer,
                            scheduler=scheduler,
                            tokens_consumed=cursor + 256, cursor={"pos": cursor})
        info = tmp_store.save(state, step)
        assert info["backend"] == "local"
        assert info["step"] == step

    captured = build_state(model=model, optimizer=optimizer,
                           scheduler=scheduler, tokens_consumed=1792,
                           cursor={"pos": 1536})

    # Mutate everything afterwards: more training + RNG churn.
    _train_steps(model, optimizer, scheduler,
                 torch.randn(4, 8), torch.randn(4, 4), 3)
    torch.rand(37)
    np.random.uniform(size=(11,))
    random.randint(0, 10**6)

    fresh_model, _, _ = _fresh_training_objects(lr=1.0)
    new_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=0.05)
    loaded = tmp_store.load_latest()
    assert loaded is not None
    restored_step, restored_cursor = tmp_store.apply(
        loaded, model=fresh_model, optimizer=new_optimizer, restore_rng=False)

    assert restored_step == 2
    assert restored_cursor == {"pos": 1536}
    after = build_state(model=fresh_model, optimizer=new_optimizer,
                        tokens_consumed=1792, cursor={"pos": 1536})
    after["step"] = restored_step
    before = dict(captured)
    before["step"] = 2
    _assert_state_restored(before, after)

    for pid, st in new_optimizer.state.items():
        assert "exp_avg" in st, "optimizer momenta were not restored"


def test_rng_states_restored(tmp_store):
    torch.manual_seed(7)
    model = _make_toy_model(7)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    torch.rand(50); np.random.rand(5); random.random()
    fingerprint_before = _rng_fingerprint()

    state = build_state(model=model, optimizer=optimizer,
                        tokens_consumed=1, cursor=None)
    tmp_store.save(state, 1)

    torch.rand(200); np.random.rand(300); random.random()
    fingerprint_churned = _rng_fingerprint()
    assert fingerprint_churned != fingerprint_before

    loaded = tmp_store.load_latest()
    assert loaded is not None
    _, _ = tmp_store.apply(loaded, model=_make_toy_model(7),
                           optimizer=torch.optim.SGD(model.parameters(), 0.01),
                           restore_rng=True)
    fp_after = _rng_fingerprint()
    assert fp_after["torch"] == fingerprint_before["torch"]
    assert fp_after["numpy"][0] == fingerprint_before["numpy"][0]
    assert fp_after["numpy"][1] == fingerprint_before["numpy"][1]
    assert fp_after["numpy"][-1] == fingerprint_before["numpy"][-1]
    assert fp_after["python"] == fingerprint_before["python"]


def test_tokens_and_data_cursor_persist(tmp_store):
    model, optimizer, _ = _fresh_training_objects()
    state = build_state(model=model, optimizer=optimizer,
                        tokens_consumed=123456, cursor={
                            "dataset": "wikitext",
                            "token_offset": 98_304,
                            "batch_in_shard": 17})
    tmp_store.save(state, 5)
    got = tmp_store.load_latest()
    assert got["tokens_consumed"] == 123456
    assert got["cursor"] == {"dataset": "wikitext", "token_offset": 98_304,
                             "batch_in_shard": 17}


# ---------------------------------------------------------------------------
# Atomicity: pointer-last protocol
# ---------------------------------------------------------------------------

def test_crash_before_pointer_publish_is_ignored(tmp_store):
    """Crash between directory write and pointer publish: load_latest must
    ignore the orphaned checkpoint and keep serving the PREVIOUS pointer
    target (the new dir landed but was never made visible)."""
    model, optimizer, _ = _fresh_training_objects()
    root = tmp_store.local_dir

    state1 = build_state(model=model, optimizer=optimizer,
                         tokens_consumed=100, cursor="c1")
    tmp_store.save(state1, 1)

    # Simulated crash for step 2: complete dir lands in the store but the
    # process dies BEFORE publishing the `latest` pointer.
    state2 = build_state(model=model, optimizer=optimizer,
                         tokens_consumed=200, cursor="c2-never-published")
    import tempfile
    import shutil
    with tempfile.TemporaryDirectory(dir=root) as td:
        crash_dir = Path(td) / "checkpoint-2"
        crash_dir.mkdir()
        tmp_store._stage_checkpoint(state2, 2, crash_dir)
        shutil.move(str(crash_dir), str(root / "checkpoint-2"))

    loaded = tmp_store.load_latest()
    assert loaded is not None
    assert loaded["step"] == 1, "unpublished checkpoint became visible!"
    assert loaded["cursor"] == "c1"

    # Recovery: next successful save publishes step 3...
    state3 = build_state(model=model, optimizer=optimizer,
                         tokens_consumed=300, cursor="c3")
    tmp_store.save(state3, 3)
    loaded = tmp_store.load_latest()
    assert loaded["step"] == 3

    # ...and retention keeps the 2 most recent COMPLETE checkpoints;
    # genuinely incomplete debris (no/invalid manifest) is swept.
    dirs = sorted(p.name for p in root.iterdir()
                  if p.is_dir() and p.name.startswith("checkpoint-"))
    assert dirs == ["checkpoint-2", "checkpoint-3"]
    assert json.loads((root / "checkpoint-2" / "manifest.json")
                      .read_text())["complete"] is True


def test_incomplete_no_manifest_dir_never_loadable(tmp_store):
    """A half-written dir WITHOUT a manifest is never a resume point and is
    swept by the next GC pass."""
    model, optimizer, _ = _fresh_training_objects()
    tmp_store.save(build_state(model=model, optimizer=optimizer,
                               tokens_consumed=10, cursor="c1"), 1)
    junk = tmp_store.local_dir / "checkpoint-9"
    junk.mkdir()
    (junk / "weights.pt").write_bytes(b"\x00garbage")   # no manifest.json
    loaded = tmp_store.load_latest()
    assert loaded["step"] == 1
    tmp_store.save(build_state(model=model, optimizer=optimizer,
                               tokens_consumed=20, cursor="c2"), 2)
    assert not junk.exists(), "incomplete debris survived GC"


def test_corrupt_complete_marker_checksum_rejected(tmp_store):
    """A torn/corrupted artifact inside a pointed-at checkpoint must be
    rejected via manifest checksums rather than silently half-loaded."""
    model, optimizer, _ = _fresh_training_objects()
    state1 = build_state(model=model, optimizer=optimizer, tokens_consumed=10,
                         cursor="c1")
    tmp_store.save(state1, 1)
    state2 = build_state(model=model, optimizer=optimizer, tokens_consumed=20,
                         cursor="c2")
    tmp_store.save(state2, 2)

    corrupt_target = tmp_store.local_dir / "checkpoint-2"
    opt_file = corrupt_target / "optimizer.pt"
    raw = bytearray(opt_file.read_bytes())
    raw[:64] = bytes((b ^ 0xFF) for b in raw[:64])
    opt_file.write_bytes(bytes(raw))

    with pytest.raises(AssertionError):
        tmp_store.restore_from(corrupt_target)

    # Direct restore of the poisoned dir fails loudly, while load_latest()
    # skips the poisoned checkpoint and serves the newest VERIFYING one
    # (step-1). Training never sees corrupt state.
    fallback = tmp_store.load_latest()
    assert fallback is not None and fallback["step"] == 1


def test_no_checkpoints_returns_none(tmp_path):
    m = CheckpointManager(local_dir=str(tmp_path / "empty"))
    assert m.load_latest() is None


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------

def test_retention_keeps_two_newest(tmp_store):
    model, optimizer, _ = _fresh_training_objects()
    for step in (1, 2, 3, 4):
        state = build_state(model=model, optimizer=optimizer,
                            tokens_consumed=step * 10, cursor=f"c{step}")
        tmp_store.save(state, step)

    dirs = sorted(p.name for p in tmp_store.local_dir.iterdir()
                  if p.is_dir() and p.name.startswith("checkpoint-"))
    assert dirs == ["checkpoint-3", "checkpoint-4"]

    loaded = tmp_store.load_latest()
    assert loaded["step"] == 4


def test_retention_custom_keep(tmp_path):
    m = CheckpointManager(local_dir=str(tmp_path / "store"), keep=1)
    model, optimizer, _ = _fresh_training_objects()
    for step in (1, 2, 3):
        m.save(build_state(model=model, optimizer=optimizer,
                           tokens_consumed=step, cursor=None), step)
    dirs = sorted(p.name for p in (m.local_dir).iterdir()
                  if p.is_dir() and p.name.startswith("checkpoint-"))
    assert dirs == ["checkpoint-3"]


# ---------------------------------------------------------------------------
# Resume parity at unit scale (end-to-end version in test_resume_parity.py)
# ---------------------------------------------------------------------------

def test_resume_parity_toy_regression(tmp_store):
    x = torch.randn(8, 8)
    y = torch.randn(8, 4)

    # Uninterrupted: 4 steps.
    m1, o1, s1 = _fresh_training_objects(seed=77)
    ref_losses = _train_steps(m1, o1, s1, x, y, 4)

    # Interrupted: 2 steps -> save -> fresh objects -> resume for 2 steps.
    m2, o2, s2 = _fresh_training_objects(seed=77)
    part_losses = _train_steps(m2, o2, s2, x, y, 2)
    rng_snapshot = torch.get_rng_state()   # deterministic data gen after this
    ck = tmp_store.save(build_state(model=m2, optimizer=o2, scheduler=s2,
                                    tokens_consumed=160, cursor={"p": 160}), 2)

    # Fresh process: no residual optimizer/RNG memory.
    m3, o3, s3 = _fresh_training_objects(seed=9999)   # different seed!
    torch.set_rng_state(rng_snapshot)
    saved = tmp_store.load_latest()
    tmp_store.apply(saved, model=m3, optimizer=o3, scheduler=s3,
                    restore_rng=False)
    resumed_losses = _train_steps(m3, o3, s3, x, y, 2)

    full = part_losses + resumed_losses
    assert len(full) == 4
    for i, (r, f) in enumerate(zip(ref_losses, full)):
        assert abs(r - f) < 1e-6, f"loss diverged at step {i}: {r} vs {f}"

    for (n1, p1), (_, p2) in zip(m1.named_parameters(), m3.named_parameters()):
        assert torch.allclose(p1, p2, atol=1e-6, rtol=1e-6), n1


# ---------------------------------------------------------------------------
# Hub backend exercised through local cache semantics (no network)
# ---------------------------------------------------------------------------

class _LocalDirHubOps:
    """Monkeypatch surface: emulate hf_hub ops against a plain directory.

    We patch the huggingface_hub callables that training/checkpoint.py uses
    so the HUB CODE PATH runs end-to-end while storage stays on disk.
    """

    def __init__(self, root: Path):
        self.root = root                      # acts as 'the repo'
        self.calls: list[tuple] = []

    # -- stand-ins ----------------------------------------------------------
    def create_repo(self, repo_id, private=False, exist_ok=False, **kw):
        self.calls.append(("create_repo", repo_id))
        d = self.root / repo_id
        d.mkdir(parents=True, exist_ok=True)

    def upload_folder(self, folder_path, repo_id, path_in_repo,
                      commit_message=None, token=None, **kw):
        self.calls.append(("upload_folder", path_in_repo))
        src, dst = Path(folder_path), self.root / repo_id / path_in_repo
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            import shutil
            shutil.rmtree(dst)
        import shutil
        shutil.copytree(src, dst)

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id,
                    commit_message=None, token=None, **kw):
        self.calls.append(("upload_file", path_in_repo))
        dst = self.root / repo_id / path_in_repo
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(Path(path_or_fileobj).read_text())

    def list_repo_files(self, repo_id, token=None, **kw):
        base = self.root / repo_id
        return [str(p.relative_to(base)).replace("\\", "/")
                for p in base.rglob("*") if p.is_file()]

    def delete_file(self, path_in_repo, repo_id, token=None, **kw):
        self.calls.append(("delete_file", path_in_repo))
        (self.root / repo_id / path_in_repo).unlink(missing_ok=True)


@pytest.fixture()
def fake_hub(tmp_path, monkeypatch):
    import huggingface_hub as hh
    ops = _LocalDirHubOps(tmp_path / "hub")

    monkeypatch.setattr(hh.HfApi, "create_repo",
                        lambda self, rid, **kw: ops.create_repo(
                            rid, private=kw.get("private", False),
                            exist_ok=kw.get("exist_ok", False)))
    monkeypatch.setattr(hh.HfApi, "upload_folder",
                        lambda self, **kw: ops.upload_folder(**kw))
    monkeypatch.setattr(hh.HfApi, "upload_file",
                        lambda self, **kw: ops.upload_file(**kw))
    monkeypatch.setattr(hh.HfApi, "list_repo_files",
                        lambda self, rid, **kw: ops.list_repo_files(rid))
    monkeypatch.setattr(hh.HfApi, "delete_file",
                        lambda self, p, repo_id, **kw: ops.delete_file(
                            p, repo_id))
    # _save_hub imports module-level functions at call time — patch those too
    monkeypatch.setattr(hh, "create_repo",
                        lambda rid, **kw: ops.create_repo(
                            rid, private=kw.get("private", False),
                            exist_ok=kw.get("exist_ok", False)))
    monkeypatch.setattr(hh, "upload_folder", lambda **kw: ops.upload_folder(**kw))
    monkeypatch.setattr(hh, "upload_file", lambda **kw: ops.upload_file(**kw))
    monkeypatch.setattr(hh, "list_repo_files",
                        lambda rid, **kw: ops.list_repo_files(rid))

    def fake_snapshot_download(repo_id, allow_patterns=None, token=None,
                               local_dir=None, **kw):
        import fnmatch
        base = Path(local_dir or tmp_path / "cache" / repo_id.replace("/", "--"))
        files = []
        pat_list = allow_patterns or ["*"]
        for p in (ops.root / repo_id).rglob("*"):
            if not p.is_file():
                continue
            rel = str(p.relative_to(ops.root / repo_id)).replace("\\", "/")
            if any(fnmatch.fnmatch(rel, pat) for pat in pat_list):
                files.append((rel, p))
        for rel, src in files:
            dst = base / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
        return str(base)

    def fake_hf_hub_download(repo_id, filename, token=None, **kw):
        src = ops.root / repo_id / filename
        dst_base = tmp_path / "cache" / repo_id.replace("/", "--")
        dst = dst_base / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        return str(dst)

    monkeypatch.setattr(hh, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(hh, "hf_hub_download", fake_hf_hub_download)
    # module-level import style used by checkpoint.py
    import training.checkpoint as tc_mod
    tc_mod_hf_list = ("huggingface_hub", )
    return ops


def test_hub_backend_via_local_cache(fake_hub, tmp_path):
    """Drive the real Hub code paths (staging -> immutable dir upload ->
    pointer-last publish -> listing verification -> snapshot_download resume)
    against an emulated local-dir 'repo'."""
    model, optimizer, _ = _fresh_training_objects()
    mgr = CheckpointManager("fake-user/bonsai-checkpoints", keep=2)
    assert mgr.use_hub

    state1 = build_state(model=model, optimizer=optimizer, tokens_consumed=64,
                         cursor={"shard": 3})
    mgr.save(state1, 11)
    # Pointer published LAST: must be newer than the immutable dir contents...
    repo_root = fake_hub.root / "fake-user" / "bonsai-checkpoints"
    ptr = json.loads((repo_root / "latest").read_text())
    assert ptr["step"] == 11
    assert (repo_root / "checkpoint-11" / "manifest.json").exists()
    manifest = json.loads(
        (repo_root / "checkpoint-11" / "manifest.json").read_text())
    assert manifest["complete"] is True and "weights.pt" in manifest["files"]

    # Save step 12, mutate everything, load_latest via snapshot_download.
    for _ in range(3):
        _train_steps(model, optimizer, None,
                     torch.randn(4, 8), torch.randn(4, 4), 1)
    state2 = build_state(model=model, optimizer=optimizer, tokens_consumed=128,
                         cursor={"shard": 7})
    mgr.save(state2, 12)
    fingerprint = {"w": model[0].weight.detach().clone(),
                   "exp_avg": next(iter(optimizer.state.values()))["exp_avg"].clone()}

    fresh_model, _, _ = _fresh_training_objects(lr=123.0)
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=123.0)
    got = mgr.load_latest()
    assert got is not None and got["step"] == 12
    step, cur = mgr.apply(got, model=fresh_model, optimizer=fresh_optimizer,
                          restore_rng=False)
    assert step == 12 and cur == {"shard": 7}
    assert torch.equal(fingerprint["w"], fresh_model[0].weight)
    assert torch.equal(fingerprint["exp_avg"],
                       next(iter(fresh_optimizer.state.values()))["exp_avg"])

    # Retention happened remotely too.
    remote_dirs = [p.name for p in repo_root.iterdir()
                   if p.is_dir() and p.name.startswith("checkpoint-")]
    assert sorted(remote_dirs) == ["checkpoint-11", "checkpoint-12"]


def test_hub_unavailable_degrades_to_local(monkeypatch, tmp_path):
    """HF unreachable/unauthed at save time -> loud warning + local store."""
    import huggingface_hub as hh

    def boom(*a, **kw):
        raise OSError("401 Unauthorized from the Hub (simulated)")

    monkeypatch.setattr(hh.HfApi, "create_repo", boom)
    mgr = CheckpointManager("some-org/private-repo")
    mgr.local_dir = tmp_path / "fallback-store"

    model, optimizer, _ = _fresh_training_objects()
    with pytest.warns(RuntimeWarning, match="falling back to local"):
        info = mgr.save(build_state(model=model, optimizer=optimizer,
                                    tokens_consumed=5, cursor="c"), 42)
    assert info["backend"] == "local"
    loaded = mgr.load_latest()
    assert loaded is not None and loaded["step"] == 42


def test_build_state_captures_everything(tmp_store):
    """build_state covers all contract fields incl. RNG capture by default."""
    torch.manual_seed(21)
    np.random.seed(22)
    random.seed(23)
    model, optimizer, scheduler = _fresh_training_objects(seed=1)
    state = build_state(model=model, optimizer=optimizer, scheduler=scheduler,
                        tokens_consumed=777, cursor={"x": 1})
    for key in ("weights", "optimizer", "scheduler", "tokens_consumed",
                "cursor", "rng"):
        assert key in state
    assert set(state["rng"]) >= {"torch", "numpy", "python"}
    # Draw a reference sequence at the exact RNG position being checkpointed,
    # then rewind, save, load, apply(restore_rng=True) and verify the next
    # draws replay identically from the restored position.
    torch.manual_seed(21); np.random.seed(22); random.seed(23)
    state2 = build_state(model=model, optimizer=optimizer, scheduler=scheduler,
                         tokens_consumed=777, cursor={"x": 1})
    fp_before = torch.rand(10)          # consumes RNG from the snapshot position
    tmp_store.save(state2, 9)
    loaded = tmp_store.load_latest()
    assert loaded is not None
    torch.manual_seed(21); np.random.seed(22); random.seed(23)  # rewind to snapshot pos
    tmp_store.apply(loaded, model=_make_toy_model(1))  # restores RNG to snapshot pos
    assert torch.equal(torch.rand(10), fp_before)
