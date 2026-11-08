#!/usr/bin/env bash
# Fetch upstream Kronos at the pre-registered commit into vendor/Kronos (gitignored).
set -euo pipefail
SHA=67b630e67f6a18c9e9be918d9b4337c960db1e9a
DIR="$(cd "$(dirname "$0")/.." && pwd)/vendor/Kronos"
if [ -d "$DIR/.git" ] && [ "$(git -C "$DIR" rev-parse HEAD)" = "$SHA" ]; then echo "vendor/Kronos already at $SHA"; exit 0; fi
rm -rf "$DIR" && mkdir -p "$DIR"
git -C "$DIR" init -q
git -C "$DIR" fetch -q --depth 1 https://github.com/shiyu-coder/Kronos.git "$SHA"
git -C "$DIR" checkout -q FETCH_HEAD
echo "vendor/Kronos at $(git -C "$DIR" rev-parse --short HEAD)"
