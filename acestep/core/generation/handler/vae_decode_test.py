"""Unit tests for extracted VAE decode mixins."""

import unittest

import torch

from acestep.core.generation.handler.vae_decode import VaeDecodeMixin
from acestep.core.generation.handler.vae_decode_chunks import VaeDecodeChunksMixin


class _DecodeOutput:
    """Minimal decoder output wrapper exposing ``sample``."""

    def __init__(self, sample: torch.Tensor):
        """Store decoded sample tensor."""
        self.sample = sample


class _FakeVae:
    """Simple VAE stub with injectable decode behavior."""

    def __init__(self, decode_fn=None):
        """Bind optional decode function and initialize default parameter."""
        self._decode_fn = decode_fn
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        """Yield a single parameter to emulate module parameter iteration."""
        yield self._param

    def cpu(self):
        """Return self to emulate in-place module migration."""
        return self

    def float(self):
        """Return self to emulate in-place dtype cast."""
        return self

    def to(self, target):
        """Return self for chained ``to(...)`` transitions in tests."""
        _ = target
        return self

    def decode(self, latents: torch.Tensor):
        """Decode latents using injected behavior or default upsample stub."""
        if self._decode_fn is not None:
            return _DecodeOutput(self._decode_fn(latents))
        bsz, _channels, latent_frames = latents.shape
        return _DecodeOutput(torch.ones(bsz, 2, latent_frames * 2))


class _DecodeHost(VaeDecodeMixin):
    """Host stub for testing VaeDecodeMixin orchestration behavior."""

    def __init__(self):
        """Initialize deterministic decode host state."""
        self.use_mlx_vae = False
        self.mlx_vae = None
        self.device = "mps"
        self.disable_tqdm = True
        self.recorded = {}

    def _get_auto_decode_chunk_size(self):
        """Return deterministic chunk size used by default path."""
        return 64

    def _should_offload_wav_to_cpu(self):
        """Return deterministic offload policy used by default path."""
        return False

    def _tiled_decode_inner(self, latents, chunk_size, overlap, offload_wav_to_cpu):
        """Record routed args and return sentinel audio tensor."""
        _ = latents
        self.recorded["chunk_size"] = chunk_size
        self.recorded["overlap"] = overlap
        self.recorded["offload"] = offload_wav_to_cpu
        return torch.ones(1, 2, 8)

    def _tiled_decode_cpu_fallback(self, latents):
        """Return fallback tensor for failure-path assertions."""
        _ = latents
        return torch.full((1, 2, 8), 2.0)

    def _mlx_vae_decode(self, latents):
        """Return MLX sentinel tensor for MLX path assertions."""
        _ = latents
        return torch.full((1, 2, 6), 3.0)


class _ChunksHost(VaeDecodeChunksMixin):
    """Host stub for testing chunk decode implementations."""

    def __init__(self):
        """Initialize default dependencies and call counters."""
        self.disable_tqdm = True
        self.vae = _FakeVae()
        self.empty_cache_calls = 0
        self.decode_on_cpu_calls = 0
        self.recorded = {}

    def _empty_cache(self):
        """Track cache-empty calls to validate OOM paths."""
        self.empty_cache_calls += 1

    def _decode_on_cpu(self, latents):
        """Return sentinel tensor for CPU-fallback assertions."""
        self.decode_on_cpu_calls += 1
        bsz = latents.shape[0]
        return torch.full((bsz, 2, 7), 9.0)


class VaeDecodeMixinTests(unittest.TestCase):
    """Verify decode orchestrator paths, fallback policy, and error propagation."""

    def test_tiled_decode_reduces_mps_chunk_and_overlap(self):
        """MPS path clamps chunk/overlap to safe configured limits."""
        host = _DecodeHost()
        out = host.tiled_decode(torch.zeros(1, 4, 128), chunk_size=64, overlap=16)
        self.assertEqual(host.recorded["chunk_size"], 32)
        self.assertEqual(host.recorded["overlap"], 8)
        self.assertFalse(host.recorded["offload"])
        self.assertEqual(tuple(out.shape), (1, 2, 8))

    def test_tiled_decode_mps_runtime_failure_uses_cpu_fallback(self):
        """MPS runtime failures fallback to CPU decode helper."""
        host = _DecodeHost()

        def _raise(*args, **kwargs):
            """Simulate runtime failure inside tiled decode implementation."""
            _ = args, kwargs
            raise RuntimeError("mps decode failure")

        host._tiled_decode_inner = _raise
        out = host.tiled_decode(torch.zeros(1, 4, 32), chunk_size=32, overlap=8)
        self.assertTrue(torch.equal(out, torch.full((1, 2, 8), 2.0)))

    def test_tiled_decode_uses_mlx_fast_path_when_available(self):
        """MLX decode should short-circuit before PyTorch path when enabled."""
        host = _DecodeHost()
        host.use_mlx_vae = True
        host.mlx_vae = object()
        out = host.tiled_decode(torch.zeros(1, 4, 32), chunk_size=32, overlap=8)
        self.assertTrue(torch.equal(out, torch.full((1, 2, 6), 3.0)))

    def test_tiled_decode_falls_back_when_mlx_decode_fails(self):
        """MLX decode errors should fallback to normal tiled decode path."""
        host = _DecodeHost()
        host.use_mlx_vae = True
        host.mlx_vae = object()

        def _mlx_raise(_latents):
            """Raise MLX failure to exercise fallback path."""
            raise ValueError("mlx failed")

        host._mlx_vae_decode = _mlx_raise
        out = host.tiled_decode(torch.zeros(1, 4, 32), chunk_size=32, overlap=8)
        self.assertEqual(tuple(out.shape), (1, 2, 8))
        self.assertEqual(host.recorded["chunk_size"], 32)

    def test_tiled_decode_non_mps_runtime_error_is_raised(self):
        """Non-MPS runtime errors should bubble to caller unchanged."""
        host = _DecodeHost()
        host.device = "cuda"

        def _raise(*args, **kwargs):
            """Raise runtime failure for non-MPS path assertion."""
            _ = args, kwargs
            raise RuntimeError("cuda decode failure")

        host._tiled_decode_inner = _raise
        with self.assertRaises(RuntimeError):
            host.tiled_decode(torch.zeros(1, 4, 32), chunk_size=32, overlap=8)


class VaeDecodeChunksMixinTests(unittest.TestCase):
    """Verify critical chunk decode paths and OOM fallback chain semantics."""

    def test_batch_sequential_decode_for_multi_sample_input(self):
        """Batch size > 1 should decode per sample then concatenate."""
        host = _ChunksHost()
        latents = torch.zeros(2, 4, 6)
        out = host._tiled_decode_inner(latents, chunk_size=10, overlap=2, offload_wav_to_cpu=False)
        self.assertEqual(tuple(out.shape), (2, 2, 12))

    def test_direct_decode_for_short_latents(self):
        """Short latents should take direct decode path without tiling loop."""
        host = _ChunksHost()
        latents = torch.zeros(1, 4, 6)
        out = host._tiled_decode_inner(latents, chunk_size=10, overlap=2, offload_wav_to_cpu=False)
        self.assertEqual(tuple(out.shape), (1, 2, 12))

    def test_overlap_adjustment_reduces_invalid_overlap(self):
        """Invalid overlap should be reduced until stride becomes positive."""
        host = _ChunksHost()

        def _capture_gpu(latents, stride, overlap, num_steps):
            """Capture overlap argument passed to GPU decode path."""
            _ = latents, stride, num_steps
            host.recorded["overlap"] = overlap
            return torch.ones(1, 2, 4)

        host._tiled_decode_gpu = _capture_gpu
        out = host._tiled_decode_inner(torch.zeros(1, 4, 10), chunk_size=4, overlap=3, offload_wav_to_cpu=False)
        self.assertEqual(host.recorded["overlap"], 1)
        self.assertEqual(tuple(out.shape), (1, 2, 4))

    def test_oom_fallback_gpu_to_offload_path(self):
        """GPU OOM should fallback to offload path before full CPU fallback."""
        host = _ChunksHost()

        def _gpu_oom(*args, **kwargs):
            """Raise OOM to force GPU fallback chain."""
            _ = args, kwargs
            raise torch.cuda.OutOfMemoryError("gpu oom")

        def _offload_ok(*args, **kwargs):
            """Return sentinel tensor from offload path."""
            _ = args, kwargs
            return torch.ones(1, 2, 5)

        host._tiled_decode_gpu = _gpu_oom
        host._tiled_decode_offload_cpu = _offload_ok
        out = host._tiled_decode_inner(torch.zeros(1, 4, 20), chunk_size=8, overlap=2, offload_wav_to_cpu=False)
        self.assertEqual(tuple(out.shape), (1, 2, 5))
        self.assertEqual(host.decode_on_cpu_calls, 0)

    def test_oom_fallback_chain_reaches_decode_on_cpu(self):
        """Repeated OOMs should end at full CPU decode fallback."""
        host = _ChunksHost()

        def _oom(*args, **kwargs):
            """Raise OOM for all tiled decode branches."""
            _ = args, kwargs
            raise torch.cuda.OutOfMemoryError("oom")

        host._tiled_decode_gpu = _oom
        host._tiled_decode_offload_cpu = _oom
        out = host._tiled_decode_inner(torch.zeros(1, 4, 20), chunk_size=8, overlap=2, offload_wav_to_cpu=False)
        self.assertTrue(torch.equal(out, torch.full((1, 2, 7), 9.0)))
        self.assertEqual(host.decode_on_cpu_calls, 1)


if __name__ == "__main__":
    unittest.main()
