import os
import torch
import torch.distributed as dist
import time
import numpy as np
from .model_tp_cuda import (
    cuda_host_register,
    cuda_host_unregister,
    cuda_host_get_device_pointer,
    cuda_device_get_attribute,
    CUDA_HOST_REGISTER_PORTABLE,
    CUDA_HOST_REGISTER_MAPPED,
    CUDA_DEV_ATTR_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM,
)
from ..ext import exllamav3_ext as ext
from multiprocessing import shared_memory
from ..util import log_tp

GLOBALS_SIZE = 128*1024
SHBUF_SIZE = 16 * 1024 ** 2
_nccl_fp32 = os.environ.get("EXL3_TP_NCCL_FP32", "0") != "0"
_no_p2p = os.environ.get("EXL3_TP_NO_P2P", "0") != "0"
_p2p_trace = os.environ.get("EXL3_TP_P2P_TRACE", "0") != "0"
_p2p_fp32 = os.environ.get("EXL3_TP_P2P_FP32", "0") != "0"
P2P_SETUP_TIMEOUT_MS = 60000
# 17 slots (16 devices + accumulator) x 2MB: 8 ring stages of the 256KB reduce chunk size
SHBUF_SIZE_R = 17 * 8 * 256 * 1024
SHBUF_SIZE_S = 16 * 1024
SHBUF_SIZE_LL = 16 * 1024
# MAX_CPU_REDUCE = SHBUF_SIZE_R // 17 // 256 * 256

class TPBackend:

    def __init__(self):
        pass

    def close(self):
        pass

    def fwd_barrier(self):
        raise NotImplementedError()


class TPBackendNull:
    """
    Collective-free stand-in used by Model.warmup() in TP mode: every rank walks its module list
    on its own and compiles/tunes its kernels concurrently with the others, never waiting in a
    collective (a rank cold-compiling Triton kernels can hold a real reduce past its sync deadline).
    Reductions leave the local partial sums in place and the output gather leaves the output
    buffer untouched: the values are garbage but finite, which is all a warmup pass needs.
    """

    def close(self):
        pass

    def fwd_barrier(self):
        pass

    def broadcast(self, tensor: torch.Tensor, src_device: int):
        pass

    def all_reduce(self, tensor: torch.Tensor, contribution: bool = True):
        pass

    def gather(self, tensor, out_tensor, gather_devices, output_device, ldims):
        pass

    def gather_small(self, tensor, out_tensor, gather_devices, output_device, ldims):
        pass

    def run_cpu_reduce_jobs(self):
        pass

    def end_cpu_reduce_jobs(self):
        pass


class TPBackendNCCL:

    def __init__(
        self,
        device: int,
        active_devices: list[int],
        output_device: int,
        init_method: str,
        master: bool,
        uuid: str,
        shbuf_size: int = SHBUF_SIZE,
    ):
        """
        NCCL-backed tensor-parallel communication backend.

        CUDA worker processes join a torch.distributed NCCL process group for barriers and all-reduce operations.
        The CPU helper process skips NCCL initialization. Operations not currently implemented directly with NCCL,
        such as broadcast and gather variants, delegate to a native shared-memory fallback backend so the rest of
        the TP code can use one backend interface.
        """
        self.device = device
        if device < 0:
            log_tp(device, f"NCCL init: skip CPU process")
            return

        self.active_devices = active_devices
        self.world_size = len(active_devices)
        self.rank = active_devices.index(device)

        log_tp(device, f"NCCL init: world_size {self.world_size}, rank {self.rank}, device {device}, init_method {init_method}")
        print(f" -- NCCL init: world_size {self.world_size}, rank {self.rank}, device {device}, init_method {init_method}")
        dist.init_process_group(
            "nccl",
            rank = self.rank,
            world_size = self.world_size,
            init_method = init_method,
        )
        self.mp_warmup_nccl(device)
        self.fallback = TPBackendNative(
            device,
            active_devices,
            output_device,
            init_method,
            master,
            uuid,
            shbuf_size
        )


    def mp_warmup_nccl(self, device):
        """
        NCCL does lazy initialization which causes the first reduction operation to take an exceedingly long time
        (20+ seconds). This seems to lead to race conditions or timeouts if it happens during a forward pass. Called
        by TP loader as soon as processes are spawned and process group is initialized.
        """
        print(f" -- NCCL warmup, device {device}, please wait...")
        x = torch.ones((6,), device = device)
        dist.all_reduce(x)
        print(f" -- Finished NCCL warmup, device {device}")


    def close(self):
        if self.device < 0:
            log_tp(self.device, f"NCCL close: skip CPU process")
            return

        dist.barrier()
        self.fallback.close()
        dist.destroy_process_group()


    def fwd_barrier(self):
        dist.barrier()


    def broadcast(self, tensor: torch.Tensor, src_device: int):
        self.fallback.broadcast(tensor, src_device)
        # src_rank = self.active_devices.index(src_device)
        # dist.broadcast(tensor, src = src_rank)


    def all_reduce(self, tensor: torch.Tensor, contribution: bool = True):
        # fp32 payloads normally go over a bf16 wire like the native backend; EXL3_TP_NCCL_FP32=1
        # reduces them in fp32 (A/B testing of the wire rounding)
        if tensor.dtype == torch.float32 and not _nccl_fp32:
            temp = tensor.to(torch.bfloat16)
            dist.all_reduce(temp, async_op = False)
            temp = temp.to(torch.float32)
            tensor.copy_(temp)
        else:
            dist.all_reduce(tensor, async_op = False)


    def gather(
        self,
        tensor: torch.Tensor,
        out_tensor: torch.Tensor | None,
        gather_devices: torch.Tensor | None,
        out_device: int,
        ldims: list[int]
    ):
        self.fallback.gather(tensor, out_tensor, gather_devices, out_device, ldims)

        # dst_rank = self.active_devices.index(out_device)
        # d_ldims = [0] * (max(self.active_devices) + 1)
        # for d, m in zip(gather_devices, ldims):
        #     d_ldims[d] = m
        # ldims = [d_ldims[d] for d in self.active_devices]
        #
        # if self.rank == dst_rank:
        #     od = 0
        #     for src, ldim in enumerate(ldims):
        #         if ldim == 0:
        #             continue
        #         out_slice = out_tensor[..., od : od + ldim]
        #         od += ldim
        #         if src == self.rank:
        #             out_slice.copy(tensor)
        #         else:
        #             # print(f"rank {self.rank} recv {out_slice.shape[-1]} from {src}")
        #             rbuf = torch.empty_like(out_slice)
        #             dist.recv(rbuf, src = src)
        #             out_slice.copy_(rbuf)
        # elif tensor.shape[-1] > 0:
        #     # print(f"rank {self.rank} send {tensor.shape[-1]} to {dst_rank}")
        #     dist.send(tensor, dst = dst_rank)


    def gather_small(
        self,
        tensor: torch.Tensor,
        out_tensor: torch.Tensor | None,
        gather_devices: torch.Tensor | None,
        out_device: int,
        ldims: list[int]
    ):
        self.fallback.gather_small(tensor, out_tensor, gather_devices, out_device, ldims)


    def run_cpu_reduce_jobs(self):
        pass


    def end_cpu_reduce_jobs(self):
        pass


class TPBackendNative:

    def __init__(
        self,
        device: int,
        active_devices: list[int],
        output_device: int,
        init_method: str,
        master: bool,
        uuid: str,
        shbuf_size: int = SHBUF_SIZE,
        cpu: bool = False
    ):
        """
        Native shared-memory tensor-parallel communication backend.

        The master process creates five named shared-memory regions and all other workers open them by UUID:
        G stores global synchronization state for the custom process-group primitives, B is the main bulk transfer
        buffer for large broadcast/gather payloads, R is reserved for CPU-assisted all-reduce staging, and S is a
        small buffer for tiny gathers. LL is dedicated to low-latency broadcasts so consumers can acknowledge and
        exit without blocking other collectives. CUDA workers register these buffers as pinned host memory.

        all_reduce() currently routes through pg_all_reduce_cpu(): GPU workers publish their contributions into the
        R buffer, a designated CPU helper performs the reduction over host memory, and workers copy the reduced
        result back. This avoids relying on NCCL for the native backend, at the cost of PCIe traffic and CPU work.
        """

        self.uuid = uuid
        self.shm_g_name = uuid + "_g"
        self.shm_b_name = uuid + "_b"
        self.shm_r_name = uuid + "_r"
        self.shm_s_name = uuid + "_s"
        self.shm_ll_name = uuid + "_ll"
        self.device = device
        self.max_num_devices = max(active_devices) + 1
        self.active_devices = active_devices
        self.shbuf_size = shbuf_size
        self.master = master
        self.cpu = cpu
        self.cpu_is_pinned = False

        size_g = GLOBALS_SIZE
        size_b = self.shbuf_size
        size_r = SHBUF_SIZE_R
        size_s = SHBUF_SIZE_S
        size_ll = SHBUF_SIZE_LL

        if master:
            log_tp(device, f"Creating SHMs")
            self.shm_g = shared_memory.SharedMemory(create = True, size = size_g, name = self.shm_g_name)
            log_tp(device, f"Created SHM: {self.shm_g_name}, {size_g} bytes")
            self.shm_b = shared_memory.SharedMemory(create = True, size = size_b, name = self.shm_b_name)
            log_tp(device, f"Created SHM: {self.shm_b_name}, {size_b} bytes")
            self.shm_r = shared_memory.SharedMemory(create = True, size = size_r, name = self.shm_r_name)
            log_tp(device, f"Created SHM: {self.shm_r_name}, {size_r} bytes")
            self.shm_s = shared_memory.SharedMemory(create = True, size = size_s, name = self.shm_s_name)
            log_tp(device, f"Created SHM: {self.shm_s_name}, {size_s} bytes")
            self.shm_ll = shared_memory.SharedMemory(create = True, size = size_ll, name = self.shm_ll_name)
            log_tp(device, f"Created SHM: {self.shm_ll_name}, {size_ll} bytes")
            self.buf_g = np.ndarray((size_g,), dtype = np.uint8, buffer = self.shm_g.buf)
            self.buf_b = np.ndarray((size_b,), dtype = np.uint8, buffer = self.shm_b.buf)
            self.buf_r = np.ndarray((size_r,), dtype = np.uint8, buffer = self.shm_r.buf)
            self.buf_s = np.ndarray((size_s,), dtype = np.uint8, buffer = self.shm_s.buf)
            self.buf_ll = np.ndarray((size_ll,), dtype = np.uint8, buffer = self.shm_ll.buf)
            self.buf_g[:] = 0
            self.buf_b[: size_b: 4096] = 0
            self.buf_r[:] = 0
            self.buf_s[:] = 0
            self.buf_ll[:] = 0
        else:
            self.shm_g = None
            self.shm_b = None
            self.shm_r = None
            self.shm_s = None
            self.shm_ll = None
            deadline = time.time() + 15
            log_tp(device, f"Opening SHMs")
            first_fnf = True
            while True:
                try:
                    if self.shm_g is None:
                        self.shm_g = shared_memory.SharedMemory(name = self.shm_g_name)
                        log_tp(device, f"Opened SHM {self.shm_g_name}")
                    if self.shm_b is None:
                        self.shm_b = shared_memory.SharedMemory(name = self.shm_b_name)
                        log_tp(device, f"Opened SHM {self.shm_b_name}")
                    if self.shm_r is None:
                        self.shm_r = shared_memory.SharedMemory(name = self.shm_r_name)
                        log_tp(device, f"Opened SHM {self.shm_r_name}")
                    if self.shm_s is None:
                        self.shm_s = shared_memory.SharedMemory(name = self.shm_s_name)
                        log_tp(device, f"Opened SHM {self.shm_s_name}")
                    if self.shm_ll is None:
                        self.shm_ll = shared_memory.SharedMemory(name = self.shm_ll_name)
                        log_tp(device, f"Opened SHM {self.shm_ll_name}")
                    break
                except FileNotFoundError:
                    if first_fnf:
                        log_tp(device, f"Waiting for SHM to appear")
                        first_fnf = False
                    if time.time() > deadline:
                        log_tp(device, f"Timeout opening SHM")
                        raise TimeoutError("Timeout waiting for master process to create SHM")
                    time.sleep(0.05)

        # Create local tensors/flags
        if self.device >= 0:
            self.abort_flag = torch.zeros((1,), device = self.device, dtype = torch.int)
        else:
            self.abort_flag = None

        # Create pinned, shared tensors
        def get_local_tensor(shm_buf, _buffer_size):
            np_view = np.ndarray(
                shape = (_buffer_size,),
                dtype = np.uint8,
                buffer = shm_buf,
                offset = 0,
            )
            return torch.as_tensor(np_view)
        self.tensor_g = get_local_tensor(self.shm_g.buf, size_g)
        self.tensor_b = get_local_tensor(self.shm_b.buf, size_b)
        self.tensor_r = get_local_tensor(self.shm_r.buf, size_r)
        self.tensor_s = get_local_tensor(self.shm_s.buf, size_s)
        self.tensor_ll = get_local_tensor(self.shm_ll.buf, size_ll)
        self.ptr_g = self.tensor_g.data_ptr()
        self.ptr_b = self.tensor_b.data_ptr()
        self.ptr_r = self.tensor_r.data_ptr()
        self.ptr_s = self.tensor_s.data_ptr()
        self.ptr_ll = self.tensor_ll.data_ptr()
        # Register the shared regions as pinned, mapped host memory and get the device-side aliases to pass to
        # kernels. On Linux desktop the alias equals the host pointer, but under WDDM (native Windows, and
        # potentially WSL2) the host pointer is not directly usable in kernels and the alias must be used instead.
        # The CPU helper process keeps the host pointers; it never launches kernels.
        self.dev_g = self.ptr_g
        self.dev_b = self.ptr_b
        self.dev_r = self.ptr_r
        self.dev_s = self.ptr_s
        self.dev_ll = self.ptr_ll
        if not self.cpu:
            def register(name, ptr, nbytes):
                log_tp(device, f"Host register {name}")
                cuda_host_register(ptr, nbytes, flags = CUDA_HOST_REGISTER_PORTABLE | CUDA_HOST_REGISTER_MAPPED)
                try:
                    dev_ptr = cuda_host_get_device_pointer(ptr)
                except RuntimeError as e:
                    raise RuntimeError(
                        f"Tensor-parallel shared buffer {name} ({nbytes} bytes) was pinned but could not be mapped "
                        f"for GPU access. The native TP collectives require GPU-mappable shared host memory, which "
                        f"this platform/driver does not provide for this region."
                    ) from e
                if dev_ptr != ptr:
                    log_tp(device, f"Host register {name}: device alias {hex(dev_ptr)} != host ptr {hex(ptr)}")
                return dev_ptr

            if self.device >= 0:
                attr = cuda_device_get_attribute(CUDA_DEV_ATTR_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM, self.device)
                log_tp(device, f"canUseHostPointerForRegisteredMem = {attr}")

            self.dev_g = register("G", self.ptr_g, self.tensor_g.numel())
            self.dev_b = register("B", self.ptr_b, self.tensor_b.numel())
            self.dev_r = register("R", self.ptr_r, self.tensor_r.numel())
            self.dev_s = register("S", self.ptr_s, self.tensor_s.numel())
            self.dev_ll = register("LL", self.ptr_ll, self.tensor_ll.numel())

        # Init global context
        if master:
            log_tp(device, f"Initializing global context")
            ext.pg_init_context(self.ptr_g)


    def close(self):
        if not self.cpu:
            log_tp(self.device, f"Host unregister G")
            cuda_host_unregister(self.ptr_g)
            log_tp(self.device, f"Host unregister B")
            cuda_host_unregister(self.ptr_b)
            log_tp(self.device, f"Host unregister R")
            cuda_host_unregister(self.ptr_r)
            log_tp(self.device, f"Host unregister S")
            cuda_host_unregister(self.ptr_s)
            log_tp(self.device, f"Host unregister LL")
            cuda_host_unregister(self.ptr_ll)
        self.shm_g.close()
        log_tp(self.device, f"Closed {self.shm_g_name}")
        self.shm_b.close()
        log_tp(self.device, f"Closed {self.shm_b_name}")
        self.shm_r.close()
        log_tp(self.device, f"Closed {self.shm_r_name}")
        self.shm_s.close()
        log_tp(self.device, f"Closed {self.shm_s_name}")
        self.shm_ll.close()
        log_tp(self.device, f"Closed {self.shm_ll_name}")
        if self.master:
            log_tp(self.device, f"Master unlink G")
            self.shm_g.unlink()
            log_tp(self.device, f"Master unlink B")
            self.shm_b.unlink()
            log_tp(self.device, f"Master unlink R")
            self.shm_r.unlink()
            log_tp(self.device, f"Master unlink S")
            self.shm_s.unlink()
            log_tp(self.device, f"Master unlink LL")
            self.shm_ll.unlink()


    def fwd_barrier(self):
        ext.pg_barrier(self.ptr_g, self.dev_g, self.active_devices, self.device, self.abort_flag)


    def broadcast(self, tensor: torch.Tensor, src_device: int):
        if tensor.numel() * tensor.element_size() <= 2048:
            ext.pg_broadcast_ll(
                self.ptr_g,
                self.dev_g,
                self.active_devices,
                self.device,
                src_device,
                tensor,
                self.dev_ll,
                SHBUF_SIZE_LL,
                self.abort_flag
            )
        else:
            ext.pg_broadcast(
                self.ptr_g,
                self.dev_g,
                self.active_devices,
                self.device,
                src_device,
                tensor,
                self.dev_b,
                self.shbuf_size,
                self.abort_flag
            )


    def all_reduce(self, tensor: torch.Tensor, contribution: bool = True):
        # if tensor.numel() * 2 < MAX_CPU_REDUCE:
        ext.pg_all_reduce_cpu(
            self.ptr_g,
            self.dev_g,
            self.active_devices,
            self.device,
            self.active_devices[0],
            tensor,
            contribution,
            self.dev_r,
            SHBUF_SIZE_R,
            self.master,
            self.abort_flag
        )
        # else:
        #     ext.pg_all_reduce(
        #         self.ptr_g,
        #         self.dev_g,
        #         self.active_devices,
        #         self.device,
        #         self.active_devices[0],
        #         tensor,
        #         self.dev_b,
        #         self.shbuf_size,
        #         self.abort_flag
        #     )


    def gather(
        self,
        tensor: torch.Tensor,
        out_tensor: torch.Tensor | None,
        gather_devices: torch.Tensor | None,
        out_device: int,
        ldims: list[int]
    ):
        if out_device == self.device:
            assert out_tensor is not None, \
                f"Gather: Output device must supply output tensor"
            assert out_tensor.shape[-1] == sum(ldims), \
                f"Gather: Output tensor must match size of concatenated slices: {sum(ldims)}"

        ext.pg_gather(
            self.ptr_g,
            self.dev_g,
            gather_devices,
            self.device,
            out_device,
            tensor,
            out_tensor,
            ldims,
            self.dev_b,
            self.shbuf_size,
            self.abort_flag
        )


    def gather_small(
        self,
        tensor: torch.Tensor,
        out_tensor: torch.Tensor | None,
        gather_devices: torch.Tensor | None,
        out_device: int,
        ldims: list[int]
    ):
        if out_device == self.device:
            assert out_tensor is not None, \
                f"Gather small: Output device must supply output tensor"
            assert out_tensor.shape[-1] == sum(ldims), \
                f"Gather small: Output tensor must match size of concatenated slices: {sum(ldims)}"

        ext.pg_gather_small(
            self.ptr_g,
            self.dev_g,
            gather_devices,
            self.device,
            out_device,
            tensor,
            out_tensor,
            ldims,
            self.dev_s,
            SHBUF_SIZE_S,
            self.abort_flag
        )


    def run_cpu_reduce_jobs(self):
        # if not self.cpu_is_pinned:
        #     set_process_priority_and_affinity()
        #     self.cpu_is_pinned = True
        ext.run_cpu_reduce_jobs(
            self.ptr_g,
            self.ptr_r,
            SHBUF_SIZE_R,
        )


    def end_cpu_reduce_jobs(self):
        if self.master:
            ext.end_cpu_reduce_jobs(
                self.ptr_g,
            )


class TPBackendP2P(TPBackendNative):

    def __init__(
        self,
        device: int,
        active_devices: list[int],
        output_device: int,
        init_method: str,
        master: bool,
        uuid: str,
        shbuf_size: int = SHBUF_SIZE,
        cpu: bool = False,
        slot_size: int = 64 * 1024 ** 2,
    ):
        """
        Native backend with the all-reduce done directly between GPUs.

        Everything the native backend does stays as it is (shared-memory regions, CPU helper process, broadcast,
        gather, barrier). Only all_reduce() changes: each CUDA rank allocates a landing arena in its own device
        memory and exports it to the other ranks over CUDA IPC; a reduce is then one kernel per rank that pushes
        its payload into the peers' arenas and sums what the peers pushed here (see all_reduce_p2p.cuh). The
        wire formats and rounding are the CPU helper's (fp16 wire for fp16, bf16 wire for bf16 and fp32 payloads,
        fp32 accumulate, one rounding), so two-rank results are bit-identical to the native backend's.
        EXL3_TP_P2P_FP32=1 reduces fp32 payloads over an exact fp32 wire instead.

        Setup is a handshake through the G region: every rank publishes its IPC handle and a verdict for each of
        three stages (arena exported, peers mapped, probe reduce correct). Any rank failing any stage, or
        EXL3_TP_NO_P2P=1, makes every rank fall back to the CPU-assisted reduce, so the ranks always agree on
        the path. Payloads larger than one landing slot also take the CPU-assisted path, which stays live for
        that reason.
        """
        super().__init__(device, active_devices, output_device, init_method, master, uuid, shbuf_size, cpu)
        self.use_p2p = False
        self.slot_size = slot_size
        self.arena = 0
        self.peer_arenas = {}
        self.n_p2p = 0
        self.n_fallback = 0
        self.output_device = output_device
        if self.cpu or self.device < 0:
            # The helper process keeps running the CPU reduce loop regardless of the verdict: with P2P active it
            # only ever sees the end-of-pass marker, and it stays available for oversize payloads
            return
        self.rank = active_devices.index(device)
        self.ranks_devices = list(active_devices)
        self.p2p_reason = None
        self._p2p_setup()
        if self.use_p2p:
            print(f" -- TP: direct P2P all-reduce active on device {device}")
        else:
            print(f" -- TP: P2P all-reduce unavailable on device {device} ({self.p2p_reason}), using CPU-assisted reduce")


    def _p2p_verdict(self, stage: int, ok: bool) -> bool:
        """
        Publish this rank's verdict for a setup stage and wait for every CUDA rank's. Non-master ranks wait for the
        master's verdict first, so nothing is published before the master has initialized the shared context.
        """
        if not self.master:
            v = ext.pg_p2p_flag_wait(self.ptr_g, stage, [self.output_device], P2P_SETUP_TIMEOUT_MS)
            if not v[0]:
                self.p2p_reason = f"stage {stage}: timeout waiting for master"
                return False
        ext.pg_p2p_flag_set(self.ptr_g, stage, self.device, 1 if ok else 2)
        values = ext.pg_p2p_flag_wait(self.ptr_g, stage, self.active_devices, P2P_SETUP_TIMEOUT_MS)
        if any(v == 0 for v in values):
            self.p2p_reason = f"stage {stage}: timeout waiting for ranks"
            return False
        if any(v != 1 for v in values):
            if ok:
                bad = [d for d, v in zip(self.active_devices, values) if v != 1]
                self.p2p_reason = f"stage {stage}: failed on device(s) {bad}"
            return False
        return True


    def _p2p_setup(self):
        ok = True
        if len(self.active_devices) < 2:
            ok = False
            self.p2p_reason = "single rank"
        elif _no_p2p:
            ok = False
            self.p2p_reason = "EXL3_TP_NO_P2P set"
        else:
            for peer in self.active_devices:
                if peer != self.device and not ext.pg_p2p_can_access(self.device, peer):
                    ok = False
                    self.p2p_reason = f"no peer access {self.device} -> {peer}"
                    break
        if ok:
            try:
                size = ext.pg_p2p_arena_size(len(self.active_devices), self.slot_size)
                self.arena = ext.pg_p2p_arena_create(self.device, size)
                ext.pg_p2p_publish(self.ptr_g, self.device, self.arena)
                log_tp(self.device, f"P2P arena created, {size} bytes")
            except RuntimeError as e:
                ok = False
                self.p2p_reason = f"arena export failed: {e}"
        if not self._p2p_verdict(0, ok):
            self._p2p_teardown()
            return

        # Map every peer's arena
        try:
            for peer in self.active_devices:
                if peer == self.device:
                    continue
                self.peer_arenas[peer] = ext.pg_p2p_open(self.ptr_g, self.device, peer)
                log_tp(self.device, f"P2P arena of device {peer} mapped")
        except RuntimeError as e:
            ok = False
            self.p2p_reason = f"peer mapping failed: {e}"
        if not self._p2p_verdict(1, ok):
            self._p2p_teardown()
            return

        # Probe: a reduce of a deterministic pattern, checked against the host. Every rank reaches this point
        # or none does, so the kernels' cross-rank waits are matched
        self.arena_list = [self.arena if d == self.device else self.peer_arenas[d] for d in self.active_devices]
        try:
            ok = self._p2p_probe()
        except RuntimeError as e:
            ok = False
            self.p2p_reason = f"probe failed: {e}"
        if not self._p2p_verdict(2, ok):
            self._p2p_teardown()
            return
        self.use_p2p = True


    def _p2p_probe(self) -> bool:
        n = 4096
        idx = torch.arange(n, dtype = torch.float32)
        def pattern(r):
            return ((idx * (r + 1) + r * 7) % 251) * 0.5 - 30.0
        expected = torch.zeros(n, dtype = torch.float32)
        for d in sorted(self.active_devices):
            expected += pattern(self.active_devices.index(d))
        expected = expected.half()
        mine = pattern(self.rank).half()
        x = torch.empty(n, dtype = torch.half, device = self.device)
        # Three passes so the landing-slot ring wraps at least once; the reduce is in place, so refill each time
        for _ in range(3):
            x.copy_(mine)
            self._p2p_all_reduce(x)
            torch.cuda.synchronize(self.device)
            if self.abort_flag.item():
                self.p2p_reason = "probe timed out"
                return False
            if not torch.equal(x.cpu(), expected):
                bad = (x.cpu() != expected).sum().item()
                self.p2p_reason = f"probe mismatch ({bad}/{n} elements)"
                return False
        return True


    def _p2p_teardown(self):
        if self.peer_arenas:
            for peer, ptr in self.peer_arenas.items():
                ext.pg_p2p_close(self.device, ptr)
            self.peer_arenas = {}
        if self.arena:
            ext.pg_p2p_arena_free(self.device, self.arena)
            self.arena = 0


    def _p2p_all_reduce(self, tensor: torch.Tensor):
        ext.pg_all_reduce_p2p(
            self.ptr_g,
            self.dev_g,
            self.arena_list,
            self.ranks_devices,
            self.rank,
            tensor,
            self.slot_size,
            _p2p_fp32,
            self.abort_flag
        )


    def _p2p_fits(self, tensor: torch.Tensor) -> bool:
        if not tensor.is_contiguous() or tensor.numel() % 8:
            return False
        wire = 4 if (tensor.dtype == torch.float32 and _p2p_fp32) else 2
        return tensor.numel() * wire <= self.slot_size


    def all_reduce(self, tensor: torch.Tensor, contribution: bool = True):
        if self.use_p2p and self._p2p_fits(tensor):
            self.n_p2p += 1
            self._p2p_all_reduce(tensor)
        else:
            self.n_fallback += 1
            super().all_reduce(tensor, contribution)


    def close(self):
        if _p2p_trace and self.device >= 0:
            print(f" -- TP: device {self.device}: {self.n_p2p} P2P reduces, {self.n_fallback} CPU-assisted reduces", flush = True)
        if self.use_p2p or self.peer_arenas or self.arena:
            # Unmap peers everywhere before anyone frees an exported arena
            for peer, ptr in self.peer_arenas.items():
                ext.pg_p2p_close(self.device, ptr)
            self.peer_arenas = {}
            ext.pg_p2p_flag_set(self.ptr_g, 3, self.device, 1)
            ext.pg_p2p_flag_wait(self.ptr_g, 3, self.active_devices, 5000)
            if self.arena:
                ext.pg_p2p_arena_free(self.device, self.arena)
                self.arena = 0
            self.use_p2p = False
        super().close()
