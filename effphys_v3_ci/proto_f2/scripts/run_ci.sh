#!/usr/bin/env bash
set -euo pipefail
mkdir -p build
iverilog -g2012 -Wall -s tb_winograd_f2x2_3x3_int8 \
  -o build/f2_sim.vvp rtl/winograd_f2x2_3x3_int8.sv tb/tb_winograd_f2x2_3x3_int8.sv
vvp build/f2_sim.vvp | tee build/iverilog_run.log
yosys -Q -p 'read_verilog -sv rtl/winograd_f2x2_3x3_int8.sv; synth_xilinx -family xc7 -top winograd_f2x2_3x3_int8; stat' \
  | tee build/yosys_xc7_stat.log
