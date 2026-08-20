#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build4
exec > >(tee build4/STAGE4_RESULT.txt) 2>&1

echo "=== FULL 2MiB DENSE1 64b MULTI-OUTSTANDING RTL SIM ==="
iverilog -g2012 -Wall -s tb_dense1_fullsize -o build4/tb_dense_full.vvp \
  dense1_weight_streamer64.sv tb_dense1_fullsize.sv
vvp build4/tb_dense_full.vvp

echo "=== FULL-SIZE STREAMER VERILATOR LINT ==="
verilator --lint-only --timing -Wall -Wno-fatal dense1_weight_streamer64.sv tb_dense1_fullsize.sv

echo "STAGE4_ALL_PASS"
