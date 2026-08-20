#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
rm -rf build
mkdir -p build
exec > >(tee build/STAGE1_RESULT.txt) 2>&1

echo "=== TOOL VERSIONS ==="
iverilog -V 2>&1 | head -n 3 || true
verilator --version
yosys -V

echo "=== 1. EXHAUSTIVE PACKED INT8 RTL SIM ==="
iverilog -g2012 -Wall -s tb_int8_packed_mul2 -o build/tb_pack.vvp \
  int8_packed_mul2.sv tb_int8_packed_mul2.sv
vvp build/tb_pack.vvp

echo "=== 2. Q2 3x3 LANE RTL SIM ==="
iverilog -g2012 -Wall -s tb_q2_conv3x3_lane -o build/tb_q2.vvp \
  int8_packed_mul2.sv q2_conv3x3_lane.sv tb_q2_conv3x3_lane.sv
vvp build/tb_q2.vvp

echo "=== 3. DENSE1 64b MULTI-OUTSTANDING RTL SIM ==="
iverilog -g2012 -Wall -s tb_dense1_weight_streamer64 -o build/tb_dense64.vvp \
  dense1_weight_streamer64.sv tb_dense1_weight_streamer64.sv
vvp build/tb_dense64.vvp

echo "=== 4. VERILATOR LINT ==="
verilator --lint-only --timing -Wall -Wno-fatal \
  int8_packed_mul2.sv q2_conv3x3_lane.sv dense1_weight_streamer64.sv

echo "=== 5. XC7 YOSYS SYNTHESIS ==="
yosys -q -p 'read_verilog -sv int8_packed_mul2.sv; synth_xilinx -family xc7 -noiopad -top int8_packed_mul2; write_json build/pack.json; stat' 2>&1 | tee build/pack_yosys.log
yosys -q -p 'read_verilog -sv int8_packed_mul2.sv q2_conv3x3_lane.sv; synth_xilinx -family xc7 -noiopad -top q2_conv3x3_lane; write_json build/q2.json; stat' 2>&1 | tee build/q2_yosys.log
yosys -q -p 'read_verilog -sv dense1_weight_streamer64.sv; synth_xilinx -family xc7 -noiopad -top dense1_weight_streamer64; write_json build/dense64.json; stat' 2>&1 | tee build/dense64_yosys.log

python3 - <<'PY'
import json
from collections import Counter

def cells(path, top):
    d=json.load(open(path))
    c=Counter(x['type'] for x in d['modules'][top].get('cells',{}).values())
    return c

def summarize(name,c):
    lut=sum(v for k,v in c.items() if k.startswith('LUT'))
    ff=sum(v for k,v in c.items() if k.startswith('FD'))
    print(f"SYNTH {name}: DSP48E1={c.get('DSP48E1',0)} RAMB36E1={c.get('RAMB36E1',0)} RAMB18E1={c.get('RAMB18E1',0)} LUT={lut} FF={ff}")
    return lut,ff

p=cells('build/pack.json','int8_packed_mul2')
q=cells('build/q2.json','q2_conv3x3_lane')
d=cells('build/dense64.json','dense1_weight_streamer64')
summarize('PACK',p)
summarize('Q2',q)
summarize('DENSE64',d)
if p.get('DSP48E1',0) != 1:
    raise SystemExit(f"FAIL: packed multiplier mapped to {p.get('DSP48E1',0)} DSPs, expected 1")
if q.get('DSP48E1',0) != 9:
    raise SystemExit(f"FAIL: q2 3x3 lane mapped to {q.get('DSP48E1',0)} DSPs, expected 9")
if d.get('DSP48E1',0) != 0:
    raise SystemExit(f"FAIL: Dense64 control unexpectedly uses {d.get('DSP48E1',0)} DSPs")
print('XC7_RESOURCE_ASSERTIONS_PASS')
PY

echo "STAGE1_ALL_PASS"
