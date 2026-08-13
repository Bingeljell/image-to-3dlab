11 categories of fixes were needed to port from CUDA to MPS: - Notes from April - so may not be true today


1. **C++ build system** — Made `custom_rasterizer` compile without CUDA (it already had CPU code, just couldn't build)
2. **MPS tensor routing** — Transparently move MPS tensors to CPU for the C++ rasterizer
3. **Device plumbing** — Replaced 10+ hardcoded `"cuda"` references with dynamic device from ComfyUI
4. **HuggingFace cache patching** — Copy local fixed Python files over HF-downloaded ones before model loading
5. **float64 guards** — MPS doesn't support float64; convert numpy arrays to float32 before moving to device
6. **Chunked attention** — Implemented memory-efficient attention (Rabe & Staats 2021) in pure PyTorch for sequences >8192 tokens, avoiding 170GB O(n²) allocations that crash MPS
7. **torch.isin() fix** — Use Python set-based approach on MPS to avoid O(N×M) memory explosion
8. **Scheduler fix** — Use DDIM instead of Euler Ancestral; diffusers 0.37.1 breaks Euler with `rescale_betas_zero_snr` (sigma=4096 at first step)
9. **n_pbr shape fix** — Handle diffusers 0.37.1 attention output shape changes for PBR materials
10. **Bugfix** — Removed erroneous `enable_model_cpu_offload()` calls on UNet (pipeline method, not model method)
11. **Install automation** — Created `install.py` for auto-building C++ extensions