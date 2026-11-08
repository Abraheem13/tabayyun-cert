"""Verify the RTX 5080 is usable (Blackwell needs sm_120 in the arch list)."""
import torch

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("arch list:", torch.cuda.get_arch_list())
    print("sm_120 supported:", "sm_120" in torch.cuda.get_arch_list())
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    import time
    t = time.time()
    for _ in range(20):
        y = x @ x
    torch.cuda.synchronize()
    print(f"bf16 matmul TFLOP/s: {20 * 2 * 4096**3 / (time.time() - t) / 1e12:.1f}")
    print("total memory GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
