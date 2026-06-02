#!/bin/bash
# Build script for aka-rk3588
# Cross-compiles on PC using aarch64-linux-gnu toolchain.
# All third-party .so dependencies (libuvc, libusb-1.0, libturbojpeg, librknnrt)
# are resolved at runtime on the board - no host .so files required.
#
# Usage:
#   ./build_rk3588.sh                    # Release
#   ./build_rk3588.sh -b Debug           # Debug
#   ./build_rk3588.sh -b Debug -l DEBUG  # with LOGD output

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
BUILD_TYPE=Release
LOG_LEVEL=INFO
GCC_COMPILER=${GCC_COMPILER:-aarch64-linux-gnu}

while getopts ":b:l:" opt; do
  case $opt in
    b) BUILD_TYPE=$OPTARG ;;
    l) LOG_LEVEL=$OPTARG  ;;
    *) echo "Usage: $0 [-b Debug|Release] [-l DEBUG|INFO|WARN|ERROR]"; exit 1 ;;
  esac
done

# ── Auto-detect native vs cross build ────────────────────────────────────────
ARCH=$(uname -m)
if [ "${ARCH}" = "aarch64" ]; then
    echo "  MODE       : native (running on board)"
    NATIVE=ON
    CC=gcc
    CXX=g++
else
    echo "  MODE       : cross-compile"
    NATIVE=OFF
    CC="${GCC_COMPILER}-gcc"
    CXX="${GCC_COMPILER}-g++"
fi

if ! command -v "${CXX}" >/dev/null 2>&1; then
    echo "ERROR: ${CXX} not found."
    if [ "${NATIVE}" = "OFF" ]; then
        echo "Install: sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu"
    fi
    exit 1
fi

echo "=== aka-rk3588 build ==="
echo "  BUILD_TYPE : ${BUILD_TYPE}"
echo "  LOG_LEVEL  : ${LOG_LEVEL}"
echo "  CXX        : ${CXX}"
echo ""

BUILD_DIR="${SCRIPT_DIR}/build"
mkdir -p "${BUILD_DIR}"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DLOG_LEVEL="${LOG_LEVEL}" \
    -DTARGET_SOC=rk3588 \
    -DNATIVE_BUILD="${NATIVE}"

cmake --build "${BUILD_DIR}" -- -j"$(nproc)"

echo ""
echo "=== Build done: ${BUILD_DIR}/tennis ==="
echo "    Deploy: scp ${BUILD_DIR}/tennis root@<board_ip>:/root/"
