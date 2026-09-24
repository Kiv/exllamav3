#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "all_reduce_p2p.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "context.cuh"
#include "timeout.cuh"
#include "all_reduce_cpu_avx2.h"   // atomic_ref
#include "../norm_row.cuh"
#include <thread>
#include <chrono>
#include <cstring>
#include <algorithm>

// Payload dtypes and wire formats. The wire is what crosses the bus and what every rank sums; the rounding of
// each contribution onto the wire and of the sum back to the payload dtype copies the CPU-assisted reduce
// exactly (fp16: F16C round-to-nearest-even; bf16: the +0x8000 shift), so results match it bit for bit.
#define P2P_DT_HALF 0
#define P2P_DT_BF16 1
#define P2P_DT_FLOAT 2
#define P2P_WIRE_HALF 0
#define P2P_WIRE_BF16 1
#define P2P_WIRE_FLOAT 2

// One line = 8 payload elements. 16-bit wires carry a line in one uint4, the fp32 wire in two
#define P2P_LINE_ELEMS 8

__device__ __forceinline__ float bf16_round_native(float f)
{
    // fp32 -> bf16 the way the native wire does it: add half an ulp to the bit pattern and truncate
    uint32_t u = __float_as_uint(f);
    u = (u + 0x8000u) & 0xffff0000u;
    return __uint_as_float(u);
}

template <int DT>
__device__ __forceinline__ void p2p_load_payload(const uint8_t* data, int line, float* v)
{
    if constexpr (DT == P2P_DT_HALF)
    {
        uint4 u = ((const uint4*) data)[line];
        const half2* h = (const half2*) &u;
        #pragma unroll
        for (int k = 0; k < 4; ++k) { float2 f = __half22float2(h[k]); v[2 * k] = f.x; v[2 * k + 1] = f.y; }
    }
    if constexpr (DT == P2P_DT_BF16)
    {
        uint4 u = ((const uint4*) data)[line];
        const __nv_bfloat162* h = (const __nv_bfloat162*) &u;
        #pragma unroll
        for (int k = 0; k < 4; ++k) { float2 f = __bfloat1622float2(h[k]); v[2 * k] = f.x; v[2 * k + 1] = f.y; }
    }
    if constexpr (DT == P2P_DT_FLOAT)
    {
        const float4* f4 = (const float4*) data + 2 * line;
        float4 a = f4[0], b = f4[1];
        v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
    }
}

// Round a contribution onto the wire (in fp32 representation) and pack it. Both are needed: the packed form
// goes to the peers, the rounded fp32 form is this rank's own contribution to its sum
template <int WIRE>
__device__ __forceinline__ void p2p_to_wire(float* v, uint4* w)
{
    if constexpr (WIRE == P2P_WIRE_HALF)
    {
        half2* h = (half2*) w;
        #pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            h[k] = __floats2half2_rn(v[2 * k], v[2 * k + 1]);
            float2 f = __half22float2(h[k]);
            v[2 * k] = f.x; v[2 * k + 1] = f.y;
        }
    }
    if constexpr (WIRE == P2P_WIRE_BF16)
    {
        uint16_t* h = (uint16_t*) w;
        #pragma unroll
        for (int k = 0; k < 8; ++k)
        {
            v[k] = bf16_round_native(v[k]);
            h[k] = (uint16_t) (__float_as_uint(v[k]) >> 16);
        }
    }
    if constexpr (WIRE == P2P_WIRE_FLOAT)
    {
        float* f = (float*) w;
        #pragma unroll
        for (int k = 0; k < 8; ++k) f[k] = v[k];
    }
}

template <int WIRE>
__device__ __forceinline__ void p2p_from_wire(const uint4* w, float* v)
{
    if constexpr (WIRE == P2P_WIRE_HALF)
    {
        const half2* h = (const half2*) w;
        #pragma unroll
        for (int k = 0; k < 4; ++k) { float2 f = __half22float2(h[k]); v[2 * k] = f.x; v[2 * k + 1] = f.y; }
    }
    if constexpr (WIRE == P2P_WIRE_BF16)
    {
        const uint16_t* h = (const uint16_t*) w;
        #pragma unroll
        for (int k = 0; k < 8; ++k) v[k] = __uint_as_float(((uint32_t) h[k]) << 16);
    }
    if constexpr (WIRE == P2P_WIRE_FLOAT)
    {
        const float* f = (const float*) w;
        #pragma unroll
        for (int k = 0; k < 8; ++k) v[k] = f[k];
    }
}

template <int DT, int WIRE>
__device__ __forceinline__ void p2p_store_result(uint8_t* data, int line, const float* acc)
{
    if constexpr (DT == P2P_DT_HALF)
    {
        uint4 u; half2* h = (half2*) &u;
        #pragma unroll
        for (int k = 0; k < 4; ++k) h[k] = __floats2half2_rn(acc[2 * k], acc[2 * k + 1]);
        ((uint4*) data)[line] = u;
    }
    if constexpr (DT == P2P_DT_BF16)
    {
        uint4 u; uint16_t* h = (uint16_t*) &u;
        #pragma unroll
        for (int k = 0; k < 8; ++k) h[k] = (uint16_t) (__float_as_uint(bf16_round_native(acc[k])) >> 16);
        ((uint4*) data)[line] = u;
    }
    if constexpr (DT == P2P_DT_FLOAT)
    {
        float4* f4 = (float4*) data + 2 * line;
        if constexpr (WIRE == P2P_WIRE_BF16)
        {
            // The sum was carried on a bf16 wire: round it back like the CPU helper does before returning fp32
            f4[0] = make_float4(bf16_round_native(acc[0]), bf16_round_native(acc[1]), bf16_round_native(acc[2]), bf16_round_native(acc[3]));
            f4[1] = make_float4(bf16_round_native(acc[4]), bf16_round_native(acc[5]), bf16_round_native(acc[6]), bf16_round_native(acc[7]));
        }
        else
        {
            f4[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
            f4[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
        }
    }
}

// One launch per rank per reduce. Blocks stripe the payload in lines of 8 elements; each block has its own flag
// per (src, slot), so blocks never synchronize with each other except through the done counter at the end.
// Per block: push this stripe (already rounded onto the wire) into every peer's landing slot, release the
// block flag in each peer, spin on the matching flags in local memory, then sum wire values in ascending
// device order in fp32 and round once, exactly as the CPU-assisted reduce does
template <int DT, int WIRE>
__global__ __launch_bounds__(P2P_NUM_THREADS)
void pg_all_reduce_p2p_kernel
(
    PGContext* __restrict__ ctx,
    const P2PPeerArenas peers,
    const int num_ranks,
    const int this_rank,
    uint8_t* __restrict__ data,
    const int num_lines,
    const size_t slot_size,
    uint32_t* abort_flag
)
{
    constexpr int WIRE_LINE = (WIRE == P2P_WIRE_FLOAT) ? 2 : 1;   // uint4 per line on the wire

    const int t = threadIdx.x;
    const int b = blockIdx.x;
    const int nb = gridDim.x;
    uint8_t* local = peers.arena[this_rank];
    uint32_t* seq_ctr = (uint32_t*) local;
    uint32_t* done_ctr = (uint32_t*) (local + 64);

    // Every block reads the sequence counter before any block can advance it (see the done counter below)
    __shared__ uint32_t s_seq;
    if (t == 0) s_seq = ldg_cv_u32(seq_ctr) + 1;
    __syncthreads();
    const uint32_t seq = s_seq;
    const int slot = (int) (seq % P2P_NUM_SLOTS);

    auto slot_ptr = [&] (uint8_t* arena, int src_rank)
    {
        return (uint4*) (arena + P2P_DATA_OFFSET + ((size_t) slot * num_ranks + src_rank) * slot_size);
    };
    auto flag_ptr = [&] (uint8_t* arena, int src_rank, int blk)
    {
        return (uint32_t*) (arena + P2P_FLAGS_OFFSET) + (src_rank * P2P_NUM_SLOTS + slot) * P2P_MAX_BLOCKS + blk;
    };

    // 1. Push this block's stripe, rounded onto the wire, into every peer's landing slot for this rank
    for (int line = b * P2P_NUM_THREADS + t; line < num_lines; line += nb * P2P_NUM_THREADS)
    {
        float v[P2P_LINE_ELEMS];
        uint4 w[WIRE_LINE];
        p2p_load_payload<DT>(data, line, v);
        p2p_to_wire<WIRE>(v, w);
        for (int r = 0; r < num_ranks; ++r)
        {
            if (r == this_rank) continue;
            uint4* dst = slot_ptr(peers.arena[r], this_rank) + (size_t) line * WIRE_LINE;
            #pragma unroll
            for (int j = 0; j < WIRE_LINE; ++j) dst[j] = w[j];
        }
    }

    // The barrier makes every thread's pushes visible to thread 0, whose release store then orders them ahead
    // of the flag on the bus (same pattern as the CPU-assisted reduce's send kernel)
    __syncthreads();
    if (t == 0)
    {
        for (int r = 0; r < num_ranks; ++r)
        {
            if (r == this_rank) continue;
            stg_release_sys_u32(flag_ptr(peers.arena[r], this_rank, b), seq);
        }
    }

    // 2. Wait for every peer's stripe to land here. Flags are compared by signed distance so the sequence may
    //    wrap. The flag is in local memory, so polling costs no bus traffic; timeouts use the shared sticky
    //    flag like every other collective
    if (t == 0)
    {
        uint64_t deadline = sync_deadline();
        for (int r = 0; r < num_ranks; ++r)
        {
            if (r == this_rank) continue;
            uint32_t* f = flag_ptr(local, r, b);
            uint32_t spins = 0;
            while ((int32_t) (ldg_acquire_sys_u32(f) - seq) < 0)
            {
                __nanosleep(32);
                if ((++spins & 0x3ff) == 0 && check_timeout(ctx, deadline, "pg_all_reduce_p2p_kernel"))
                {
                    *abort_flag = 1;
                    break;
                }
            }
            if (*((volatile uint32_t*) abort_flag)) break;
        }
    }
    __syncthreads();

    // 3. Sum in place: this rank's wire-rounded contribution plus every peer's landed line, accumulated in fp32
    //    in ascending device order, rounded once. Landed lines are read with cache-global loads (written into
    //    this GPU's memory by a peer, so L1 must not serve them)
    if (!*((volatile uint32_t*) abort_flag))
    {
        for (int line = b * P2P_NUM_THREADS + t; line < num_lines; line += nb * P2P_NUM_THREADS)
        {
            float mine[P2P_LINE_ELEMS];
            uint4 w[WIRE_LINE];
            p2p_load_payload<DT>(data, line, mine);
            p2p_to_wire<WIRE>(mine, w);
            // The first contribution is copied, not added to zero: 0 + (-0) would lose the sign of zero that
            // the CPU helper's memcpy-then-add sequence keeps
            float acc[P2P_LINE_ELEMS];
            for (int k = 0; k < num_ranks; ++k)
            {
                int r = peers.sum_order[k];
                float v[P2P_LINE_ELEMS];
                if (r == this_rank)
                {
                    #pragma unroll
                    for (int j = 0; j < P2P_LINE_ELEMS; ++j) v[j] = mine[j];
                }
                else
                {
                    const uint4* src = slot_ptr(local, r) + (size_t) line * WIRE_LINE;
                    uint4 lw[WIRE_LINE];
                    #pragma unroll
                    for (int j = 0; j < WIRE_LINE; ++j) lw[j] = __ldcg(src + j);
                    p2p_from_wire<WIRE>(lw, v);
                }
                #pragma unroll
                for (int j = 0; j < P2P_LINE_ELEMS; ++j) acc[j] = (k == 0) ? v[j] : acc[j] + v[j];
            }
            p2p_store_result<DT, WIRE>(data, line, acc);
        }
    }

    // 4. Last block out advances the sequence counter. Every block incremented the done counter only after
    //    reading seq, so no block can observe the new value for this launch
    __syncthreads();
    if (t == 0)
    {
        uint32_t prev = atomicInc(done_ctr, (unsigned int) nb - 1);
        if (prev == (unsigned int) nb - 1)
        {
            __threadfence();
            stg_wt_u32(seq_ctr, seq);
        }
    }
}


size_t pg_p2p_arena_size(int num_ranks, size_t slot_size)
{
    return (size_t) P2P_DATA_OFFSET + (size_t) P2P_NUM_SLOTS * num_ranks * slot_size;
}


uintptr_t pg_p2p_arena_create(int device, size_t size)
{
    const at::cuda::OptionalCUDAGuard device_guard(device);
    void* ptr = nullptr;
    // Dedicated allocation (not the caching allocator) so the IPC handle covers exactly this arena
    cuda_check(cudaMalloc(&ptr, size));
    cuda_check(cudaMemset(ptr, 0, size));
    cuda_check(cudaDeviceSynchronize());
    return (uintptr_t) ptr;
}


void pg_p2p_arena_free(int device, uintptr_t arena)
{
    const at::cuda::OptionalCUDAGuard device_guard(device);
    cudaError_t e = cudaFree((void*) arena);
    if (e != cudaSuccess) (void) cudaGetLastError();
}


bool pg_p2p_can_access(int device, int peer_device)
{
    int v = 0;
    cudaError_t e = cudaDeviceCanAccessPeer(&v, device, peer_device);
    if (e != cudaSuccess) { (void) cudaGetLastError(); return false; }
    return v != 0;
}


void pg_p2p_publish(uintptr_t ctx, int device, uintptr_t arena)
{
    static_assert(sizeof(cudaIpcMemHandle_t) == P2P_IPC_HANDLE_SIZE, "cudaIpcMemHandle_t size mismatch");
    const at::cuda::OptionalCUDAGuard device_guard(device);
    PGContext* ctx_ptr = (PGContext*) ctx;
    cudaIpcMemHandle_t handle;
    cuda_check(cudaIpcGetMemHandle(&handle, (void*) arena));
    memcpy(ctx_ptr->p2p_ipc_handle[device], &handle, sizeof(handle));
}


uintptr_t pg_p2p_open(uintptr_t ctx, int device, int peer_device)
{
    const at::cuda::OptionalCUDAGuard device_guard(device);
    PGContext* ctx_ptr = (PGContext*) ctx;
    cudaIpcMemHandle_t handle;
    memcpy(&handle, ctx_ptr->p2p_ipc_handle[peer_device], sizeof(handle));
    void* ptr = nullptr;
    cuda_check(cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess));
    return (uintptr_t) ptr;
}


void pg_p2p_close(int device, uintptr_t peer_arena)
{
    const at::cuda::OptionalCUDAGuard device_guard(device);
    cudaError_t e = cudaIpcCloseMemHandle((void*) peer_arena);
    if (e != cudaSuccess) (void) cudaGetLastError();
}


void pg_p2p_flag_set(uintptr_t ctx, int round, int device, uint32_t value)
{
    PGContext* ctx_ptr = (PGContext*) ctx;
    TORCH_CHECK(round >= 0 && round < P2P_ROUNDS, "pg_p2p_flag_set: bad round");
    atomic_ref<uint32_t> f(&ctx_ptr->p2p_flags[round][device]);
    f.store_release(value);
}


std::vector<uint32_t> pg_p2p_flag_wait(uintptr_t ctx, int round, std::vector<int> devices, int timeout_ms)
{
    PGContext* ctx_ptr = (PGContext*) ctx;
    TORCH_CHECK(round >= 0 && round < P2P_ROUNDS, "pg_p2p_flag_wait: bad round");
    std::vector<uint32_t> values(devices.size(), 0);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (true)
    {
        bool all = true;
        for (size_t i = 0; i < devices.size(); ++i)
        {
            atomic_ref<uint32_t> f(&ctx_ptr->p2p_flags[round][devices[i]]);
            values[i] = f.load_acquire();
            if (!values[i]) all = false;
        }
        if (all || std::chrono::steady_clock::now() > deadline) return values;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}


void pg_all_reduce_p2p
(
    uintptr_t ctx,
    uintptr_t ctx_dev,
    std::vector<uintptr_t> arenas,
    std::vector<int> devices,
    int this_rank,
    at::Tensor& tensor,
    size_t slot_size,
    bool fp32_wire,
    at::Tensor& abort_flag
)
{
    const int num_ranks = (int) devices.size();
    TORCH_CHECK(num_ranks >= 2 && num_ranks <= P2P_MAX_RANKS, "pg_all_reduce_p2p: bad rank count");
    TORCH_CHECK(arenas.size() == devices.size(), "pg_all_reduce_p2p: arenas/devices mismatch");
    const int this_device = devices[this_rank];

    const at::cuda::OptionalCUDAGuard device_guard(this_device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pg_check_timeout(ctx);

    uint8_t* data_ptr = (uint8_t*) tensor.data_ptr();
    const int64_t numel = tensor.numel();
    TORCH_CHECK(tensor.is_contiguous(), "pg_all_reduce_p2p: tensor must be contiguous");
    TORCH_CHECK(numel % P2P_LINE_ELEMS == 0, "pg_all_reduce_p2p: numel must be a multiple of 8");
    const int num_lines = (int) (numel / P2P_LINE_ELEMS);
    const bool wire_f32 = tensor.dtype() == at::kFloat && fp32_wire;
    const size_t wire_bytes = (size_t) numel * (wire_f32 ? 4 : 2);
    TORCH_CHECK(wire_bytes <= slot_size, "pg_all_reduce_p2p: payload exceeds landing slot");

    P2PPeerArenas peers;
    for (int r = 0; r < P2P_MAX_RANKS; ++r) { peers.arena[r] = nullptr; peers.sum_order[r] = 0; }
    for (int r = 0; r < num_ranks; ++r) peers.arena[r] = (uint8_t*) arenas[r];
    std::vector<int> order(num_ranks);
    for (int r = 0; r < num_ranks; ++r) order[r] = r;
    std::sort(order.begin(), order.end(), [&] (int a, int b) { return devices[a] < devices[b]; });
    for (int r = 0; r < num_ranks; ++r) peers.sum_order[r] = (uint8_t) order[r];

    int num_blocks = (int) CEIL_DIVIDE(num_lines, P2P_NUM_THREADS);
    num_blocks = MAX(1, MIN(num_blocks, P2P_MAX_BLOCKS));

    PGContext* ctx_d = (PGContext*) ctx_dev;
    uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();

    #define P2P_LAUNCH(DT, WIRE) \
        pg_all_reduce_p2p_kernel<DT, WIRE><<<num_blocks, P2P_NUM_THREADS, 0, stream>>> \
            (ctx_d, peers, num_ranks, this_rank, data_ptr, num_lines, slot_size, abort_flag_ptr)

    if (tensor.dtype() == at::kHalf) P2P_LAUNCH(P2P_DT_HALF, P2P_WIRE_HALF);
    else if (tensor.dtype() == at::kBFloat16) P2P_LAUNCH(P2P_DT_BF16, P2P_WIRE_BF16);
    else if (tensor.dtype() == at::kFloat && wire_f32) P2P_LAUNCH(P2P_DT_FLOAT, P2P_WIRE_FLOAT);
    else if (tensor.dtype() == at::kFloat) P2P_LAUNCH(P2P_DT_FLOAT, P2P_WIRE_BF16);
    else TORCH_CHECK(false, "pg_all_reduce_p2p: Unknown dtype");

    #undef P2P_LAUNCH
    cuda_check(cudaPeekAtLastError());
}


// Fused reduce + residual epilogue for decode-sized payloads: one block per row (token). The block first
// reduces its row exactly as pg_all_reduce_p2p_kernel would (same wire, same rounding, result written back
// into y in place), then applies the epilogue the transformer block would otherwise launch as separate
// kernels: P2P_FUSE_NORM runs rms_norm_row<RES_IN> (r += y; out = norm(r) * w, the rms_norm_res_in kernel's
// arithmetic, same block size so the reduction order is identical); P2P_FUSE_ADD does r += y. Rows are
// limited to P2P_MAX_BLOCKS by the per-block flags
#define P2P_FUSE_ADD 0
#define P2P_FUSE_NORM 1

template <int DT, int WIRE, int MODE, typename output_t, typename weight_t, typename residual_t>
__global__ __launch_bounds__(P2P_NUM_THREADS)
void pg_all_reduce_p2p_fused_kernel
(
    PGContext* __restrict__ ctx,
    const P2PPeerArenas peers,
    const int num_ranks,
    const int this_rank,
    uint8_t* __restrict__ data,          // y: (rows, dim) payload, reduced in place
    residual_t* __restrict__ r,          // residual (rows, dim), updated in place
    const weight_t* __restrict__ w,      // norm weight (dim), NORM mode
    output_t* __restrict__ out,          // normed output (rows, dim), NORM mode
    const int dim,
    const float epsilon,
    const float constant_bias,
    const float constant_scale,
    const size_t slot_size,
    uint32_t* abort_flag
)
{
    constexpr int WIRE_LINE = (WIRE == P2P_WIRE_FLOAT) ? 2 : 1;
    using payload_t = typename std::conditional<DT == P2P_DT_HALF, half, float>::type;

    const int t = threadIdx.x;
    const int row = blockIdx.x;
    const int nb = gridDim.x;
    const int lines_per_row = dim / P2P_LINE_ELEMS;
    const int line0 = row * lines_per_row;
    uint8_t* local = peers.arena[this_rank];
    uint32_t* seq_ctr = (uint32_t*) local;
    uint32_t* done_ctr = (uint32_t*) (local + 64);

    __shared__ uint32_t s_seq;
    if (t == 0) s_seq = ldg_cv_u32(seq_ctr) + 1;
    __syncthreads();
    const uint32_t seq = s_seq;
    const int slot = (int) (seq % P2P_NUM_SLOTS);

    auto slot_ptr = [&] (uint8_t* arena, int src_rank)
    {
        return (uint4*) (arena + P2P_DATA_OFFSET + ((size_t) slot * num_ranks + src_rank) * slot_size);
    };
    auto flag_ptr = [&] (uint8_t* arena, int src_rank, int blk)
    {
        return (uint32_t*) (arena + P2P_FLAGS_OFFSET) + (src_rank * P2P_NUM_SLOTS + slot) * P2P_MAX_BLOCKS + blk;
    };

    // 1. Push this row's lines
    for (int i = t; i < lines_per_row; i += blockDim.x)
    {
        int line = line0 + i;
        float v[P2P_LINE_ELEMS];
        uint4 wv[WIRE_LINE];
        p2p_load_payload<DT>(data, line, v);
        p2p_to_wire<WIRE>(v, wv);
        for (int rk = 0; rk < num_ranks; ++rk)
        {
            if (rk == this_rank) continue;
            uint4* dst = slot_ptr(peers.arena[rk], this_rank) + (size_t) line * WIRE_LINE;
            #pragma unroll
            for (int j = 0; j < WIRE_LINE; ++j) dst[j] = wv[j];
        }
    }
    __syncthreads();
    if (t == 0)
    {
        for (int rk = 0; rk < num_ranks; ++rk)
        {
            if (rk == this_rank) continue;
            stg_release_sys_u32(flag_ptr(peers.arena[rk], this_rank, row), seq);
        }
    }

    // 2. Wait for the peers' copies of this row
    if (t == 0)
    {
        uint64_t deadline = sync_deadline();
        for (int rk = 0; rk < num_ranks; ++rk)
        {
            if (rk == this_rank) continue;
            uint32_t* f = flag_ptr(local, rk, row);
            uint32_t spins = 0;
            while ((int32_t) (ldg_acquire_sys_u32(f) - seq) < 0)
            {
                __nanosleep(32);
                if ((++spins & 0x3ff) == 0 && check_timeout(ctx, deadline, "pg_all_reduce_p2p_fused_kernel"))
                {
                    *abort_flag = 1;
                    break;
                }
            }
            if (*((volatile uint32_t*) abort_flag)) break;
        }
    }
    __syncthreads();

    if (!*((volatile uint32_t*) abort_flag))
    {
        // 3. Sum into y in place (identical to the plain kernel)
        for (int i = t; i < lines_per_row; i += blockDim.x)
        {
            int line = line0 + i;
            float mine[P2P_LINE_ELEMS];
            uint4 wv[WIRE_LINE];
            p2p_load_payload<DT>(data, line, mine);
            p2p_to_wire<WIRE>(mine, wv);
            float acc[P2P_LINE_ELEMS];
            for (int k = 0; k < num_ranks; ++k)
            {
                int rk = peers.sum_order[k];
                float v[P2P_LINE_ELEMS];
                if (rk == this_rank)
                {
                    #pragma unroll
                    for (int j = 0; j < P2P_LINE_ELEMS; ++j) v[j] = mine[j];
                }
                else
                {
                    const uint4* src = slot_ptr(local, rk) + (size_t) line * WIRE_LINE;
                    uint4 lw[WIRE_LINE];
                    #pragma unroll
                    for (int j = 0; j < WIRE_LINE; ++j) lw[j] = __ldcg(src + j);
                    p2p_from_wire<WIRE>(lw, v);
                }
                #pragma unroll
                for (int j = 0; j < P2P_LINE_ELEMS; ++j) acc[j] = (k == 0) ? v[j] : acc[j] + v[j];
            }
            p2p_store_result<DT, WIRE>(data, line, acc);
        }
        __syncthreads();

        // 4. Epilogue on the reduced row
        if constexpr (MODE == P2P_FUSE_NORM)
        {
            rms_norm_row<RES_IN, payload_t, output_t, weight_t, residual_t>
                ((const payload_t*) data, w, out, r, epsilon, row, dim, constant_bias, constant_scale, 1);
        }
        else
        {
            // r += y, computed in fp32 and rounded to the residual dtype: what torch's in-place add does
            const payload_t* y = (const payload_t*) data + (size_t) row * dim;
            residual_t* rr = r + (size_t) row * dim;
            for (int c = t; c < dim; c += blockDim.x)
            {
                float yv, rv;
                if constexpr (std::is_same_v<payload_t, half>) yv = __half2float(y[c]); else yv = y[c];
                if constexpr (std::is_same_v<residual_t, half>) rv = __half2float(rr[c]); else rv = rr[c];
                float sum = rv + yv;
                if constexpr (std::is_same_v<residual_t, half>) rr[c] = __float2half_rn(sum); else rr[c] = sum;
            }
        }
    }

    // 5. Last block out advances the sequence counter
    __syncthreads();
    if (t == 0)
    {
        uint32_t prev = atomicInc(done_ctr, (unsigned int) nb - 1);
        if (prev == (unsigned int) nb - 1)
        {
            __threadfence();
            stg_wt_u32(seq_ctr, seq);
        }
    }
}


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
)
{
    const int num_ranks = (int) devices.size();
    TORCH_CHECK(num_ranks >= 2 && num_ranks <= P2P_MAX_RANKS, "pg_all_reduce_p2p_fused: bad rank count");
    TORCH_CHECK(arenas.size() == devices.size(), "pg_all_reduce_p2p_fused: arenas/devices mismatch");
    const int this_device = devices[this_rank];

    const at::cuda::OptionalCUDAGuard device_guard(this_device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pg_check_timeout(ctx);

    const int dim = (int) tensor.size(-1);
    const int64_t numel = tensor.numel();
    const int rows = (int) (numel / dim);
    TORCH_CHECK(tensor.is_contiguous() && residual.is_contiguous(), "pg_all_reduce_p2p_fused: tensors must be contiguous");
    TORCH_CHECK(dim % P2P_LINE_ELEMS == 0, "pg_all_reduce_p2p_fused: dim must be a multiple of 8");
    TORCH_CHECK(rows >= 1 && rows <= P2P_MAX_BLOCKS, "pg_all_reduce_p2p_fused: too many rows");
    TORCH_CHECK(residual.numel() == numel, "pg_all_reduce_p2p_fused: residual shape mismatch");
    const bool wire_f32 = tensor.dtype() == at::kFloat && fp32_wire;
    const size_t wire_bytes = (size_t) numel * (wire_f32 ? 4 : 2);
    TORCH_CHECK(wire_bytes <= slot_size, "pg_all_reduce_p2p_fused: payload exceeds landing slot");
    if (mode == P2P_FUSE_NORM)
    {
        TORCH_CHECK(weight.has_value() && out.has_value(), "pg_all_reduce_p2p_fused: norm mode needs weight and out");
        TORCH_CHECK(weight.value().numel() == dim, "pg_all_reduce_p2p_fused: weight shape mismatch");
        TORCH_CHECK(out.value().numel() == numel && out.value().is_contiguous(), "pg_all_reduce_p2p_fused: out shape mismatch");
    }

    P2PPeerArenas peers;
    for (int r = 0; r < P2P_MAX_RANKS; ++r) { peers.arena[r] = nullptr; peers.sum_order[r] = 0; }
    for (int r = 0; r < num_ranks; ++r) peers.arena[r] = (uint8_t*) arenas[r];
    std::vector<int> order(num_ranks);
    for (int r = 0; r < num_ranks; ++r) order[r] = r;
    std::sort(order.begin(), order.end(), [&] (int a, int b) { return devices[a] < devices[b]; });
    for (int r = 0; r < num_ranks; ++r) peers.sum_order[r] = (uint8_t) order[r];

    // Same block size as rms_norm for this row width, so the norm's reduction order matches
    const int threads = rms_norm_threads(dim);
    PGContext* ctx_d = (PGContext*) ctx_dev;
    uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
    uint8_t* data_ptr = (uint8_t*) tensor.data_ptr();
    void* r_ptr = residual.data_ptr();
    const void* w_ptr = weight.has_value() ? weight.value().data_ptr() : nullptr;
    void* o_ptr = out.has_value() ? out.value().data_ptr() : nullptr;
    auto tp = tensor.scalar_type(); auto tr = residual.scalar_type();
    auto tw = weight.has_value() ? weight.value().scalar_type() : at::kHalf;
    auto to = out.has_value() ? out.value().scalar_type() : at::kHalf;

    #define P2P_FLAUNCH(DT, WIRE, MODE, OT, WT, RT) \
        pg_all_reduce_p2p_fused_kernel<DT, WIRE, MODE, OT, WT, RT><<<rows, threads, 0, stream>>> \
            (ctx_d, peers, num_ranks, this_rank, data_ptr, (RT*) r_ptr, (const WT*) w_ptr, (OT*) o_ptr, \
             dim, epsilon, constant_bias, constant_scale, slot_size, abort_flag_ptr)
    #define P2P_FDISPATCH_RT(DT, WIRE, MODE, OT, WT) \
        if (tr == at::kFloat) P2P_FLAUNCH(DT, WIRE, MODE, OT, WT, float); \
        else if (tr == at::kHalf) P2P_FLAUNCH(DT, WIRE, MODE, OT, WT, half); \
        else TORCH_CHECK(false, "pg_all_reduce_p2p_fused: residual dtype")
    #define P2P_FDISPATCH_MODE(DT, WIRE) \
        if (mode == P2P_FUSE_ADD) { P2P_FDISPATCH_RT(DT, WIRE, P2P_FUSE_ADD, half, half); } \
        else if (tw == at::kHalf && to == at::kHalf) { P2P_FDISPATCH_RT(DT, WIRE, P2P_FUSE_NORM, half, half); } \
        else if (tw == at::kBFloat16 && to == at::kHalf) { P2P_FDISPATCH_RT(DT, WIRE, P2P_FUSE_NORM, half, bfloat16); } \
        else TORCH_CHECK(false, "pg_all_reduce_p2p_fused: weight/out dtype")

    if (tp == at::kHalf) { P2P_FDISPATCH_MODE(P2P_DT_HALF, P2P_WIRE_HALF); }
    else if (tp == at::kFloat && wire_f32) { P2P_FDISPATCH_MODE(P2P_DT_FLOAT, P2P_WIRE_FLOAT); }
    else if (tp == at::kFloat) { P2P_FDISPATCH_MODE(P2P_DT_FLOAT, P2P_WIRE_BF16); }
    else TORCH_CHECK(false, "pg_all_reduce_p2p_fused: payload dtype");

    #undef P2P_FLAUNCH
    #undef P2P_FDISPATCH_RT
    #undef P2P_FDISPATCH_MODE
    cuda_check(cudaPeekAtLastError());
}
