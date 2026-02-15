#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [android_output_dir]" >&2
  exit 1
fi

ROOT_DIR="${1:-dist/ci-android}"
REQUIRED_ABIS=(arm64-v8a armeabi-v7a x86_64)

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "Android output directory not found: $ROOT_DIR" >&2
  exit 1
fi

find_readelf() {
  if command -v llvm-readelf >/dev/null 2>&1; then
    command -v llvm-readelf
    return
  fi
  if command -v readelf >/dev/null 2>&1; then
    command -v readelf
    return
  fi
  if [[ -n "${ANDROID_NDK_HOME:-}" ]]; then
    local ndk_readelf
    ndk_readelf="$(find "${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt" -type f -name llvm-readelf 2>/dev/null | head -n 1 || true)"
    if [[ -n "$ndk_readelf" ]]; then
      echo "$ndk_readelf"
      return
    fi
  fi
  return 1
}

READELF_BIN="$(find_readelf || true)"
if [[ -z "$READELF_BIN" ]]; then
  echo "Failed to locate llvm-readelf/readelf in PATH or ANDROID_NDK_HOME." >&2
  exit 1
fi

declare -A seen_abi=()
mapfile -t so_files < <(find "$ROOT_DIR" -type f -name "*.so" | sort)

if [[ ${#so_files[@]} -eq 0 ]]; then
  echo "No .so files found under $ROOT_DIR" >&2
  exit 1
fi

echo "Using readelf tool: $READELF_BIN"
echo "Checking ${#so_files[@]} shared libraries under $ROOT_DIR"

status=0
for so_file in "${so_files[@]}"; do
  for abi in "${REQUIRED_ABIS[@]}"; do
    if [[ "$so_file" == *"/${abi}/"* ]]; then
      seen_abi["$abi"]=1
    fi
  done

  mapfile -t aligns < <("$READELF_BIN" -lW "$so_file" | awk '/^[[:space:]]*LOAD[[:space:]]/ { print tolower($NF) }')
  if [[ ${#aligns[@]} -eq 0 ]]; then
    echo "FAIL: $so_file has no LOAD program headers" >&2
    status=1
    continue
  fi

  for align in "${aligns[@]}"; do
    if [[ ! "$align" =~ ^0x0*4000$ ]]; then
      echo "FAIL: $so_file has LOAD align $align (expected 0x4000)" >&2
      status=1
    fi
  done

  soname="$("$READELF_BIN" -dW "$so_file" | awk -F'[][]' '/SONAME/ { print $2; exit }')"
  expected_soname="$(basename "$so_file")"
  if [[ -z "$soname" ]]; then
    echo "FAIL: $so_file is missing SONAME (expected $expected_soname)" >&2
    status=1
  elif [[ "$soname" != "$expected_soname" ]]; then
    echo "FAIL: $so_file has SONAME $soname (expected $expected_soname)" >&2
    status=1
  fi
done

for abi in "${REQUIRED_ABIS[@]}"; do
  if [[ -z "${seen_abi[$abi]:-}" ]]; then
    echo "FAIL: missing ABI output for $abi under $ROOT_DIR" >&2
    status=1
  fi
done

if [[ $status -ne 0 ]]; then
  exit $status
fi

echo "PASS: all Android shared libraries use 16 KB LOAD alignment (0x4000) and stable SONAME."
