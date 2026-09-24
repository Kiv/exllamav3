from __future__ import annotations
from typing_extensions import override
import torch
from ..util.tensor import to2
from ..model.config import Config
from . import Module, RMSNorm, LayerNorm, Attention, GatedDeltaNet, GatedMLP, MLP, BlockSparseMLP, Linear
from .hyperconnections import HyperConnection
from ..util import profile_opt

class TransformerBlock(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int | None = None,
        attn_norm: RMSNorm | LayerNorm | None = None,
        attn: Attention | GatedDeltaNet | None = None,
        attn_post_norm: RMSNorm | LayerNorm | None = None,
        mlp_norm: RMSNorm | LayerNorm | None = None,
        mlp: MLP | GatedMLP | BlockSparseMLP | None = None,
        mlp_post_norm: RMSNorm | LayerNorm | None = None,
        attn_hc: HyperConnection | None = None,
        mlp_hc: HyperConnection | None = None,
        key_layer_scalar: str | None = None,
        key_attn_resid_scalar: str | None = None,
        key_mlp_resid_scalar: str | None = None,
        qmap: str | None = None,
        qbits_key: str = "bits",
        out_dtype: torch.dtype = None
    ):
        super().__init__(config, key, None)

        self.layer_idx = layer_idx
        self.attn_norm = attn_norm
        self.attn = attn
        self.attn_post_norm = attn_post_norm
        self.mlp_norm = mlp_norm
        self.mlp = mlp
        self.mlp_post_norm = mlp_post_norm
        self.attn_hc = attn_hc
        self.mlp_hc = mlp_hc
        self.qbits_key = qbits_key
        self.out_dtype = out_dtype

        self.key_layer_scalar = key_layer_scalar
        self.key_attn_resid_scalar = key_attn_resid_scalar
        self.key_mlp_resid_scalar = key_mlp_resid_scalar
        self.layer_scalar_t = None
        self.layer_scalar_f = None
        self.attn_resid_scalar = None
        self.mlp_resid_scalar = None

        # Hyperconnection sites (mHC): the block's residual is (bsz, seq, hc_mult, hidden)
        # fp32 streams, mixed at each sublayer site instead of the plain residual add
        if attn_hc is not None or mlp_hc is not None:
            assert attn_hc is not None and mlp_hc is not None, \
                "hyperconnections require both attn_hc and mlp_hc"
            assert all(v is None for v in (
                attn_post_norm, mlp_post_norm,
                key_layer_scalar, key_attn_resid_scalar, key_mlp_resid_scalar,
            )), \
                "hyperconnections cannot combine with residual scalars/post-norms"

        self.register_submodule(self.attn_hc)
        self.register_submodule(self.attn_norm)
        self.register_submodule(self.attn)
        self.register_submodule(self.attn_post_norm)
        self.register_submodule(self.mlp_hc)
        self.register_submodule(self.mlp_norm)
        self.register_submodule(self.mlp)
        self.register_submodule(self.mlp_post_norm)

        self.num_slices = mlp.num_slices if mlp else 1


    @override
    def optimizer_targets(self):
        a = self.attn.optimizer_targets() if self.attn else []
        m = self.mlp.optimizer_targets() if self.mlp else []
        return [a, m]

    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        if self.key_layer_scalar:
            self.layer_scalar_t = self.config.stc.get_tensor(
                self.key + "." + self.key_layer_scalar,
                None,
                allow_bf16 = True,
                no_defer = True,
            )
            assert self.layer_scalar_t.numel() == 1
            self.layer_scalar_f = self.layer_scalar_t.float().item()

        # TODO: Residual scalar tensors could be baked into preceding modules for models that use them
        #       (currently only Step3.7 vision tower)
        if self.key_attn_resid_scalar:
            self.attn_resid_scalar = self.config.stc.get_tensor(
                self.key + "." + self.key_attn_resid_scalar,
                device,
                allow_bf16 = True,
                no_defer = True,
            )
        if self.key_mlp_resid_scalar:
            self.mlp_resid_scalar = self.config.stc.get_tensor(
                self.key + "." + self.key_mlp_resid_scalar,
                device,
                allow_bf16 = True,
                no_defer = True,
            )

    def unload(self):
        super().unload()
        self.layer_scalar_t = None
        self.attn_resid_scalar = None
        self.mlp_resid_scalar = None

    def get_tensors(self):
        t = {}
        if self.key_layer_scalar is not None:
            t[self.key + "." + self.key_layer_scalar] = self.layer_scalar_t.data.contiguous()
        if self.key_attn_resid_scalar is not None:
            t[self.key + "." + self.key_attn_resid_scalar] = self.attn_resid_scalar.data.contiguous()
        if self.key_mlp_resid_scalar is not None:
            t[self.key + "." + self.key_mlp_resid_scalar] = self.mlp_resid_scalar.data.contiguous()
        return t

    def weights_numel(self):
        return (
            super().weights_numel() +
            (1 if self.key_layer_scalar is not None else 0) +
            (self.attn_resid_scalar.numel() if self.attn_resid_scalar is not None else 0) +
            (self.mlp_resid_scalar.numel() if self.mlp_resid_scalar is not None else 0)
        )

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        export_state = params.get("export_state_layers")
        export_state = export_state and self.layer_idx in export_state and params.get("layer_instance", 0) == 0

        y_resid = None  # pending attn output whose residual add is folded into the MLP input norm
        y_fused = None  # MLP input already produced by a reduce fused with the residual add + norm

        # Tensor-parallel backends that can fuse a sublayer's all-reduce with the residual epilogue
        # (TPBackendP2P.all_reduce_fused): the sublayer parks its partial sums instead of reducing, and
        # the block reduces them together with x += y and, for attention, the MLP input norm
        backend = params.get("backend")
        fuse = (
            backend is not None and getattr(backend, "use_p2p", False) and not export_state and
            not params.get("prefill")
        )
        fuse_attn = (
            fuse and self.attn is not None and not self.attn_hc and self.attn_post_norm is None and
            self.attn_resid_scalar is None
        )
        fuse_mlp = (
            fuse and self.mlp is not None and not self.mlp_hc and self.mlp_post_norm is None and
            self.mlp_resid_scalar is None
        )

        def collect_pending(y):
            # Resolve a deferred collect that could not be fused: reduce the partial sums the way
            # the sublayer would have, then redo any dtype conversion it applied afterwards
            pending = backend.pending_collect
            backend.pending_collect = None
            if pending is None:
                return y, None
            py, contribution = pending
            if py is not y:
                backend.all_reduce(py, contribution)
                return py.to(y.dtype), None
            return y, pending

        if self.attn:
            if self.attn_hc:
                hc_post, hc_comb, y = self.attn_hc.mix(x, params)
                y = y.half()
                if self.attn_norm:
                    y = self.attn_norm.forward(y, params, out_dtype = torch.half)
            elif self.attn_norm:
                y = self.attn_norm.forward(x, params, out_dtype = torch.half)
            else:
                y = x.half()
            if fuse_attn:
                backend.defer_collect = True
            y = self.attn.forward(y, params)
            if fuse_attn:
                backend.defer_collect = False
                y, pending = collect_pending(y)
                if pending is not None:
                    norm = self.mlp_norm if (self.mlp is not None and not self.mlp_hc) else None
                    if norm is not None and backend.can_fuse_collect(y, x, norm):
                        y_fused = backend.all_reduce_fused(y, x, norm, out_dtype = torch.half)
                    elif backend.can_fuse_collect(y, x):
                        backend.all_reduce_fused(y, x)
                        y = None  # residual add done
                    else:
                        backend.all_reduce(y, pending[1])
            if params.get("prefill") and not export_state:
                return x
            if y_fused is not None or y is None:
                pass  # residual add already applied by the fused reduce
            elif self.attn_resid_scalar is not None:
                y *= self.attn_resid_scalar
            if y_fused is not None or y is None:
                pass
            elif self.attn_hc:
                x = self.attn_hc.apply_(x, y, hc_post, hc_comb, params)
            elif self.attn_post_norm:
                self.attn_post_norm.forward(y, params, residual = x)
            elif self.mlp is not None and self.mlp_norm is not None and self.mlp_norm.can_fuse_residual(x, y):
                y_resid = y
            else:
                x += y

        if self.mlp:
            if self.mlp_hc:
                hc_post, hc_comb, y = self.mlp_hc.mix(x, params)
                y = y.half()
                if self.mlp_norm:
                    y = self.mlp_norm.forward(y, params, out_dtype = torch.half)
            else:
                params["residual"] = x
                if y_fused is not None:
                    y = y_fused
                elif y_resid is not None:
                    y = self.mlp_norm.forward(y_resid, params, out_dtype = torch.half, residual_in = x)
                elif self.mlp_norm:
                    y = self.mlp_norm.forward(x, params, out_dtype = torch.half)
                else:
                    y = x.half()
            if fuse_mlp:
                backend.defer_collect = True
            y = self.mlp.forward(y, params)
            if fuse_mlp:
                backend.defer_collect = False
                y, pending = collect_pending(y)
                if pending is not None:
                    if backend.can_fuse_collect(y, x):
                        backend.all_reduce_fused(y, x)
                        y = None  # residual add done
                    else:
                        backend.all_reduce(y, pending[1])
            if y is None:
                pass
            elif self.mlp_resid_scalar is not None:
                y *= self.mlp_resid_scalar
            if y is None:
                pass
            elif self.mlp_hc:
                x = self.mlp_hc.apply_(x, y, hc_post, hc_comb, params)
            elif self.mlp_post_norm:
                self.mlp_post_norm.forward(y, params, residual = x)
            else:
                x += y

        if export_state:
            s = params.get("export_states")
            if not s:
                s = params["export_states"] = []
            # With hyperconnections the residual is a stream stack; export the stream mean as
            # the collapsed hidden state (streams start as broadcast copies of the embedding)
            x_ = x.mean(dim = 2) if self.attn_hc else x
            if x_.dtype == torch.half:
                s.append(x_.clamp_(-65504.0, 65504.0))
            else:
                x_ = x_.half()
                x_.clamp_(-65504.0, 65504.0)
                s.append(x_)

        if self.layer_scalar_f is not None:
            x *= self.layer_scalar_f

        return to2(x, out_dtype, self.out_dtype)


    def get_name(self):
        name = super().get_name()
        if not self.attn and not self.mlp:
            name += " (no-op)"
        return name


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."

        def _export(child):
            nonlocal producer
            return child.tp_export(plan, producer) if child is not None else None

        return {
            "cls": TransformerBlock,
            "kwargs": {
                "key": self.key,
                "layer_idx": self.layer_idx,
                "out_dtype": self.out_dtype,
                "key_layer_scalar": self.key_layer_scalar,
                "key_attn_resid_scalar": self.key_attn_resid_scalar,
                "key_mlp_resid_scalar": self.key_mlp_resid_scalar,
            },
            **{name: _export(getattr(self, name, None)) for name in (
                "attn_hc",
                "attn_norm",
                "attn",
                "attn_post_norm",
                "mlp_hc",
                "mlp_norm",
                "mlp",
                "mlp_post_norm",
            )},
            # Per-layer scalars load from the tensor collection, which TP children don't have
            "layer_scalar_f": self.layer_scalar_f,
            "attn_resid_scalar": producer.send(self.attn_resid_scalar),
            "mlp_resid_scalar": producer.send(self.mlp_resid_scalar),
            "device": self.device,
        }


    @staticmethod
    def tp_import(local_context, exported, plan):
        consumer = local_context["consumer"]
        device = local_context["device"]

        def _import(name):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan) \
                if exported.get(name) else None

        module = TransformerBlock(
            config = None,
            **exported["kwargs"],
            attn_hc = _import("attn_hc"),
            attn_norm = _import("attn_norm"),
            attn = _import("attn"),
            attn_post_norm = _import("attn_post_norm"),
            mlp_hc = _import("mlp_hc"),
            mlp_norm = _import("mlp_norm"),
            mlp = _import("mlp"),
            mlp_post_norm = _import("mlp_post_norm"),
        )

        module.layer_scalar_f = exported.get("layer_scalar_f")
        module.attn_resid_scalar = consumer.recv(exported.get("attn_resid_scalar"), cuda = True)
        module.mlp_resid_scalar = consumer.recv(exported.get("mlp_resid_scalar"), cuda = True)
        module.device = device
        return module


class ParallelDecoderBlock(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int | None = None,
        input_norm: RMSNorm | LayerNorm | None = None,
        attn: Attention | None = None,
        mlp: MLP | GatedMLP | None = None,
        qmap: str | None = None,
        qbits_key: str = "bits",
        out_dtype: torch.dtype = None
    ):
        super().__init__(config, key, None)

        self.layer_idx = layer_idx
        self.input_norm = input_norm
        self.attn = attn
        self.mlp = mlp
        self.qbits_key = qbits_key
        self.out_dtype = out_dtype

        self.register_submodule(self.input_norm)
        self.register_submodule(self.attn)
        self.register_submodule(self.mlp)

        self.num_slices = mlp.num_slices if mlp else 1

        self.tp_reduce = False


    @override
    def optimizer_targets(self):
        a = self.attn.optimizer_targets() if self.attn else []
        m = self.mlp.optimizer_targets() if self.mlp else []
        return [a, m]


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        y = self.input_norm.forward(x, params, out_dtype = torch.half)
        y1 = self.attn.forward(y, params)
        if not params.get("prefill"):
            y2 = self.mlp.forward(y, params)
            y1 += y2

            if self.tp_reduce:
                self.tp_collect(params["backend"], y1)

            x += y1

        return to2(x, out_dtype, self.out_dtype)


    def get_name(self):
        name = super().get_name()
        if not self.attn and not self.mlp:
            name += " (no-op)"
        return name


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."

        def _export(child):
            nonlocal producer
            return child.tp_export(plan, producer) if child is not None else None

        return {
            "cls": ParallelDecoderBlock,
            "kwargs": {
                "key": self.key,
                "layer_idx": self.layer_idx,
                "out_dtype": self.out_dtype,
            },
            **{name: _export(getattr(self, name, None)) for name in (
                "input_norm",
                "attn",
                "mlp",
            )},
            "device": self.device,
        }


    @staticmethod
    def tp_import(local_context, exported, plan):
        device = local_context["device"]

        def _import(name, **kwargs):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan, **kwargs) \
                if exported.get(name) else None

        module = ParallelDecoderBlock(
            config = None,
            **exported["kwargs"],
            input_norm = _import("input_norm"),
            attn = _import("attn", skip_reduction = True),
            mlp = _import("mlp", skip_reduction = True),
        )
        module.device = device

        # Use single reduction for sum of mlp and attn
        module.tp_reduce = True
        module.tp_owner = module.tp_single_owner(
            local_context, exported["attn"]["kwargs"]["key"], exported["mlp"]["kwargs"]["key"])
        return module
