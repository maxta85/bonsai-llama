#!/usr/bin/env bash
# Clone and build microsoft/bitnet (bitnet.cpp) — the official inference
# engine for BitNet b1.58 models (microsoft/bitnet-b1.58-2B-4T and community
# BitNet models). Provides optimized I2_S / TL1 kernels for x86 and ARM CPUs.
#
# This is the engine that actually delivers the speed/energy/latency wins
# promised by the BitNet paper. The transformers path works but gives NO
# efficiency gain (see the model card warning).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BITNET_DIR="$ROOT/third_party/bitnet.cpp"
BUILD_DIR="$ROOT/build/bitnet_cpp"

echo "Cloning microsoft/bitnet (bitnet.cpp) into $BITNET_DIR"
mkdir -p "$ROOT/third_party"
if [ -d "$BITNET_DIR/.git" ]; then
    echo "  already cloned; pulling latest"
    git -C "$BITNET_DIR" pull --ff-only
else
    git clone --depth 1 https://github.com/microsoft/bitnet "$BITNET_DIR"
fi

echo "Configuring CMake in $BUILD_DIR"
# bitnet.cpp supports CPU backends. For GPU, see the repo's docs.
cmake -B "$BUILD_DIR" -S "$BITNET_DIR" \
    -DCMAKE_BUILD_TYPE=Release

echo "Building bitnet.cpp..."
cmake --build "$BUILD_DIR" -j

echo
echo "Done. Run the official BitNet b1.58 2B model:"
echo "  # Download the GGUF variant (CPU inference):"
echo "  hf download microsoft/bitnet-b1.58-2B-4T-gguf --local-dir models/bitnet-2b"
echo "  # Run:"
echo "  $BUILD_DIR/bin/bitnet -m models/bitnet-2b/bitnet-b1.58-2b-4t.gguf -p 'Hello'"
echo
echo "Or the packed HF variant via the bitnet runner (see repo for details)."
