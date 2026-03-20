"""Copy DAgger episodes into the augmented training dataset, renumbering to avoid collisions."""
import shutil
from pathlib import Path

SRC = Path("data/dagger/chunk-000")
DST = Path("../robocasa/datasets/v1.0/pretrain/atomic/OpenCabinet/20250819/lerobot/augmented")

if not DST.exists():
    print(f"ERROR: Destination not found: {DST}")
    exit(1)

count = 0
for i in range(26):
    src_file = SRC / f"episode_{i:06d}.parquet"
    dst_file = DST / f"episode_{i + 107:06d}.parquet"
    if not src_file.exists():
        print(f"  Skipping {src_file} (not found)")
        continue
    shutil.copy2(src_file, dst_file)
    print(f"  {src_file.name} -> {dst_file.name}")
    count += 1

print(f"\nDone! Copied {count} DAgger episodes into {DST}")
