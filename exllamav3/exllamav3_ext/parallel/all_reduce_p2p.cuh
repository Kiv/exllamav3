#pragma once

#include <ATen/Tensor.h>
#include <vector>
#include <cstdint>

// Direct GPU-to-GPU (P2P) all-reduce for the native tensor-parallel backend.
//
// Each rank owns an arena in its own device memory, exported to the other ranks over CUDA IPC. A reduce is
// one kernel per rank: push the local payload into every peer's landing slot for this rank (posted PCIe/NVLink
// writes), release a per-block flag in the peer's arena, spin on the corresponding flags in the LOCAL arena,
// then sum local + landed in fp32 and round once. No host-side work per call, so the kernel is graph-capturable.
// Contributions cross the bus in the same wire format and with the same rounding as the CPU-assisted reduce
// (fp16 wire for fp16, bf16 wire for bf16 and, by default, for fp32), so results are bit-identical to it.
//
// Arena layout (bytes):
//   [0, 64)                         seq counter: number of reduces completed on this rank (device-side sequence
//                                   discovery, same convention as the CPU-assisted reduce's stage counters)
//   [64, 128)                       done counter for the last-block-out protocol that advances seq
//   [P2P_FLAGS_OFFSET, +4 KB)       flags[src_rank][slot][block], written by src_rank over the bus, read locally
//   [P2P_DATA_OFFSET, ...)          landing slots: (slot * num_ranks + src_rank) * slot_size, written by src_rank
//
// Slot reuse (P2P_NUM_SLOTS = 2) is safe without acknowledgements: a rank issues reduce N+2 only after its kernel
// N+1 completed, which required the peer's flag N+1, which the peer writes at the start of its kernel N+1, i.e.
// after its kernel N (and every read of slot N % 2) completed in stream order.

#define P2P_MAX_RANKS 16
#define P2P_NUM_SLOTS 2
#define P2P_MAX_BLOCKS 16
#define P2P_NUM_THREADS 1024
#define P2P_FLAGS_OFFSET 128
#define P2P_DATA_OFFSET 8192

struct P2PPeerArenas
{
    uint8_t* arena[P2P_MAX_RANKS];   // indexed by rank; this rank's own arena at [this_rank]
    uint8_t sum_order[P2P_MAX_RANKS];  // ranks in ascending device order, the CPU helper's accumulation order
};

// Arena lifecycle (each CUDA rank, its own device)
size_t pg_p2p_arena_size(int num_ranks, size_t slot_size);
uintptr_t pg_p2p_arena_create(int device, size_t size);
void pg_p2p_arena_free(int device, uintptr_t arena);

// IPC handle exchange through the shared PGContext (host VA)
bool pg_p2p_can_access(int device, int peer_device);
void pg_p2p_publish(uintptr_t ctx, int device, uintptr_t arena);
uintptr_t pg_p2p_open(uintptr_t ctx, int device, int peer_device);
void pg_p2p_close(int device, uintptr_t peer_arena);

// Small host-side flag rounds through the PGContext, for the setup handshake
void pg_p2p_flag_set(uintptr_t ctx, int round, int device, uint32_t value);
std::vector<uint32_t> pg_p2p_flag_wait(uintptr_t ctx, int round, std::vector<int> devices, int timeout_ms);

void pg_all_reduce_p2p
(
    uintptr_t ctx,
    uintptr_t ctx_dev,
    std::vector<uintptr_t> arenas,   // by rank
    std::vector<int> devices,        // by rank
    int this_rank,
    at::Tensor& tensor,
    size_t slot_size,
    bool fp32_wire,   // fp32 payloads: exact fp32 wire instead of the native backend's bf16 wire
    at::Tensor& abort_flag
);

// Fused reduce + residual epilogue for decode-sized payloads (rows <= P2P_MAX_BLOCKS). mode 0: residual += y;
// mode 1: residual += y, out = rmsnorm(residual) * weight (rms_norm_res_in arithmetic). y is reduced in place too.
void pg_all_reduce_p2p_fused
(
    uintptr_t ctx,
    uintptr_t ctx_dev,
    std::vector<uintptr_t> arenas,
    std::vector<int> devices,
    int this_rank,
    at::Tensor& tensor,
    at::Tensor& residual,
    c10::optional<at::Tensor> weight,
    c10::optional<at::Tensor> out,
    float epsilon,
    float constant_bias,
    float constant_scale,
    int mode,
    size_t slot_size,
    bool fp32_wire,
    at::Tensor& abort_flag
);
