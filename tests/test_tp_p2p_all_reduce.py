import os
import uuid
import pytest
import torch
import multiprocessing as mp

# Drives TPBackendP2P directly in two spawned worker processes (no model, no CPU helper): random-size reduces
# across the three payload dtypes, every result compared bit for bit with a host reference that accumulates in
# fp32 in ascending device order and rounds once, which is the contract the kernel makes with the CPU-assisted
# reduce. Enough iterations that the landing-slot ring wraps many times, and payloads up to a few MB so the
# multi-block striping and per-block flags are exercised. Skips when there are not two GPUs with peer access.

SIZES = [8, 64, 640, 5120, 5 * 5120, 65536, 262144, 1 << 20]  # elements
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
ITERS = 40
SLOT_SIZE = 8 * 1024 ** 2


def _worker(device, active_devices, output_device, uid, conn):
    try:
        torch.cuda.set_device(device)
        from exllamav3.model.model_tp_backend import TPBackendP2P
        backend = TPBackendP2P(
            device = device,
            active_devices = active_devices,
            output_device = output_device,
            init_method = "",
            master = (device == output_device),
            uuid = uid,
            slot_size = SLOT_SIZE,
        )
        if not backend.use_p2p:
            conn.send(("unavailable", backend.p2p_reason))
            backend.close()
            return
        rank = active_devices.index(device)
        order = sorted(range(len(active_devices)), key = lambda r: active_devices[r])
        failures = []
        for it in range(ITERS):
            for dtype in DTYPES:
                for n in SIZES:
                    g = torch.Generator().manual_seed(it * 100003 + n)
                    parts = [torch.randn(n, generator = g) * 8 for _ in active_devices]
                    if dtype != torch.float32:
                        parts = [p.to(dtype) for p in parts]
                    # Host reference with the native wire rules: fp32 payloads are rounded to bf16 (+0x8000
                    # shift) on the wire and the sum rounded the same way; fp16/bf16 payloads go verbatim
                    def bf16_native(t):
                        u = t.contiguous().view(torch.int32)
                        return ((u + 0x8000) & -65536).view(torch.float32)
                    contrib = [bf16_native(p) if dtype == torch.float32 else p.float() for p in parts]
                    ref = torch.zeros(n, dtype = torch.float32)
                    for r in order:
                        ref += contrib[r]
                    if dtype == torch.float32:
                        ref = bf16_native(ref)
                    elif dtype == torch.bfloat16:
                        ref = bf16_native(ref).to(torch.bfloat16)
                    else:
                        ref = ref.to(dtype)
                    x = parts[rank].to(device)
                    backend.all_reduce(x)
                    got = x.cpu()
                    if not torch.equal(got, ref):
                        bad = (got != ref).sum().item()
                        failures.append(f"iter {it} {dtype} n={n}: {bad} mismatches")
                        if len(failures) > 8:
                            break
            if backend.abort_flag.item():
                failures.append(f"abort flag set at iter {it}")
                break
        backend.close()
        conn.send(("ok", failures))
    except Exception as e:
        conn.send(("error", repr(e)))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason = "needs two GPUs")
def test_p2p_all_reduce_bit_exact():
    from exllamav3.ext import exllamav3_ext as ext
    if not (ext.pg_p2p_can_access(0, 1) and ext.pg_p2p_can_access(1, 0)):
        pytest.skip("no peer access between GPU 0 and 1")
    if os.environ.get("EXL3_TP_NO_P2P", "0") != "0":
        pytest.skip("EXL3_TP_NO_P2P set")

    ctx = mp.get_context("spawn")
    active_devices = [0, 1]
    output_device = 1
    uid = uuid.uuid4().hex
    procs, conns = [], []
    for device in active_devices:
        parent, child = ctx.Pipe()
        p = ctx.Process(target = _worker, args = (device, active_devices, output_device, uid, child))
        p.start()
        procs.append(p)
        conns.append(parent)
    results = []
    for p, c in zip(procs, conns):
        if c.poll(600):
            results.append(c.recv())
        else:
            results.append(("error", "worker timed out"))
        p.join(30)
        if p.is_alive():
            p.kill()
    for status, payload in results:
        if status == "unavailable":
            pytest.skip(f"P2P backend unavailable: {payload}")
        assert status == "ok", payload
        assert payload == [], payload
