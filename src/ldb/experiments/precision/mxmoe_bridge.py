"""Inference bridge to the patched upstream MxMoE library, default CUDA stream."""

import ctypes as C

import torch


class Bridge:
    def __init__(self, library):
        self.lib = C.CDLL(str(library))
        self.lib.ldb_mxmoe_count.restype = C.c_int
        self.lib.ldb_mxmoe_name.argtypes = [C.c_int]
        self.lib.ldb_mxmoe_name.restype = C.c_char_p
        self.lib.ldb_mxmoe_error.restype = C.c_char_p
        self.lib.ldb_mxmoe_run.argtypes = [C.c_int, C.POINTER(C.c_void_p), C.c_int]
        self.lib.ldb_mxmoe_pack.argtypes = [C.c_void_p] * 4 + [C.c_int] * 4 + [C.c_bool]
        self.lib.ldb_mxmoe_pack_codes.argtypes = [C.c_void_p] * 4 + [C.c_int] * 3
        self.names = [
            self.lib.ldb_mxmoe_name(i).decode() for i in range(self.lib.ldb_mxmoe_count())
        ]

    def check(self, status):
        if status:
            raise RuntimeError(self.lib.ldb_mxmoe_error().decode())

    def pack(self, weight, bits=4):
        if bits not in (4, 8):
            raise ValueError("Expected INT4 or INT8")
        if weight.dtype != torch.float16 or not weight.is_cuda or not weight.is_contiguous():
            raise ValueError("Expected contiguous FP16 CUDA weight")
        self.default_stream(weight.device)
        n, k = weight.shape
        packed = torch.empty(n * k // (16 // bits), device=weight.device, dtype=torch.float16)
        scale = torch.empty(n * 2, device=weight.device, dtype=torch.float16)
        fake = torch.empty_like(weight)
        self.check(
            self.lib.ldb_mxmoe_pack(
                weight.data_ptr(),
                packed.data_ptr(),
                scale.data_ptr(),
                fake.data_ptr(),
                n,
                k,
                bits,
                -1,
                False,
            )
        )
        return (packed, scale, n, k, bits), fake

    @staticmethod
    def default_stream(device):
        if torch.cuda.current_stream(device).cuda_stream != 0:
            raise ValueError("Upstream launch API currently requires the default CUDA stream")

    def pack_codes(self, codes, scale_offset, bits):
        if (
            bits not in (4, 8)
            or codes.ndim != 2
            or not codes.is_cuda
            or codes.dtype != torch.int32
            or not codes.is_contiguous()
        ):
            raise ValueError("Expected contiguous INT32 CUDA codes and 4/8 bits")
        n, k = codes.shape
        if (
            scale_offset.shape != (n, 2)
            or scale_offset.dtype != torch.float16
            or scale_offset.device != codes.device
            or not scale_offset.is_contiguous()
        ):
            raise ValueError("Expected per-channel FP16 scale/offset pairs")
        self.default_stream(codes.device)
        packed = torch.empty(n * k // (16 // bits), device=codes.device, dtype=torch.float16)
        scales = torch.empty(n * 2, device=codes.device, dtype=torch.float16)
        self.check(
            self.lib.ldb_mxmoe_pack_codes(
                codes.data_ptr(),
                scale_offset.data_ptr(),
                packed.data_ptr(),
                scales.data_ptr(),
                n,
                k,
                bits,
            )
        )
        return packed, scales, n, k, bits

    def run(self, matrices, inputs, kernel):
        return self.prepare(matrices, inputs, kernel)()

    def prepare(self, matrices, inputs, kernel):
        if not matrices or len(matrices) != len(inputs):
            raise ValueError("Expected matched nonempty groups")
        if not (0 <= kernel < len(self.names)) or any(
            f"w{m[4]}a16_g-1_asym:" not in self.names[kernel] for m in matrices
        ):
            raise ValueError("Select a kernel supporting every matrix precision")
        device = inputs[0].device
        self.default_stream(device)
        for (packed, scale, n, k, bits), x in zip(matrices, inputs):
            if (
                x.dtype != torch.float16
                or x.device != device
                or packed.device != device
                or scale.device != device
                or not x.is_contiguous()
                or x.ndim != 2
                or x.shape[1] != k
                or not x.shape[0]
            ):
                raise ValueError("Invalid grouped input")
        outputs = [
            torch.empty((x.shape[0], m[2]), device=device, dtype=torch.float16)
            for m, x in zip(matrices, inputs)
        ]

        def ptrs(values):
            return torch.tensor([v.data_ptr() for v in values], device=device, dtype=torch.int64)

        dims = torch.tensor(
            [[x.shape[0], m[2], m[3]] for m, x in zip(matrices, inputs)], dtype=torch.int32
        )
        quant = torch.tensor([[16, m[4], -1, 0] for m in matrices], dtype=torch.int32)
        stride_k = torch.tensor([m[3] for m in matrices], device=device, dtype=torch.int64)
        stride_n = torch.tensor([m[2] for m in matrices], device=device, dtype=torch.int64)
        out_ptr = ptrs(outputs)
        scales = ptrs([m[1] for m in matrices])
        args = [
            ptrs(inputs),
            ptrs([m[0] for m in matrices]),
            scales,
            scales,
            out_ptr,
            out_ptr,
            stride_k,
            stride_k,
            stride_n,
            stride_n,
            dims.to(device),
            dims,
            quant.to(device),
            quant,
        ]
        addresses = (C.c_void_p * 14)(*[t.data_ptr() for t in args])
        return PreparedCall(self, kernel, matrices, inputs, outputs, args, addresses)


class PreparedCall:
    """Own fixed input/output buffers and metadata for repeated inference launches."""

    def __init__(self, bridge, kernel, matrices, inputs, outputs, args, addresses):
        self.bridge, self.kernel = bridge, kernel
        self.matrices, self.inputs, self.outputs = matrices, inputs, outputs
        self.args, self.addresses = args, addresses

    def __call__(self):
        self.bridge.default_stream(self.inputs[0].device)
        self.bridge.check(
            self.bridge.lib.ldb_mxmoe_run(self.kernel, self.addresses, len(self.inputs))
        )
        return self.outputs
