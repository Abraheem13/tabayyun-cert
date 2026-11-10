"""Verify the RTX 5080 is usable (Blackwell needs sm_120 in the arch list).
Prints the report and saves it to results/env.txt."""
import platform
import time
from pathlib import Path

import torch

lines = [f"python {platform.python_version()} on {platform.system()} {platform.release()}",
         f"torch {torch.__version__}  cuda {torch.version.cuda}",
         f"cuda available: {torch.cuda.is_available()}"]
if torch.cuda.is_available():
    arch = torch.cuda.get_arch_list()
    lines += [f"device: {torch.cuda.get_device_name(0)}",
              f"capability: {torch.cuda.get_device_capability(0)}",
              f"arch list: {arch}",
              f"sm_120 supported: {'sm_120' in arch}",
              f"total memory GB: {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)}"]
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        x @ x
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(20):
        x @ x
    torch.cuda.synchronize()
    lines.append(f"bf16 matmul TFLOP/s: {20 * 2 * 4096**3 / (time.time() - t) / 1e12:.1f}")
else:
    lines.append("WARNING: no CUDA. Reinstall torch with: "
                 "pip install torch --index-url https://download.pytorch.org/whl/cu128")
report = "\n".join(lines)
print(report)
out = Path(__file__).resolve().parents[1] / "results" / "env.txt"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(report + "\n", encoding="utf-8")
