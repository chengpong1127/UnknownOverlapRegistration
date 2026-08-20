#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build3
exec > >(tee build3/STAGE3_RESULT.txt) 2>&1

echo "=== PIXEL ENGINE FUNCTIONAL/CYCLE RTL SIM ==="
iverilog -g2012 -Wall -s tb_pixel_engines -o build3/tb_pixel.vvp \
  int8_packed_mul2.sv p3q4_conv3x3_group.sv p4q2_conv3x3_group.sv \
  conv2_p3q4_pixel_engine.sv conv_p4q2_pixel_engine.sv tb_pixel_engines.sv
vvp build3/tb_pixel.vvp

echo "=== VERILATOR LINT ==="
verilator --lint-only --timing -Wall -Wno-fatal \
  int8_packed_mul2.sv p3q4_conv3x3_group.sv p4q2_conv3x3_group.sv \
  conv2_p3q4_pixel_engine.sv conv_p4q2_pixel_engine.sv

echo "=== XC7 SYNTHESIS OF ACCUMULATING ENGINES ==="
yosys -q -p 'read_verilog -sv int8_packed_mul2.sv p3q4_conv3x3_group.sv conv2_p3q4_pixel_engine.sv; synth_xilinx -family xc7 -noiopad -top conv2_p3q4_pixel_engine; write_json build3/conv2_engine.json; stat' 2>&1 | tee build3/conv2_yosys.log
yosys -q -p 'read_verilog -sv int8_packed_mul2.sv p4q2_conv3x3_group.sv conv_p4q2_pixel_engine.sv; synth_xilinx -family xc7 -noiopad -top conv_p4q2_pixel_engine; write_json build3/p4_engine.json; stat' 2>&1 | tee build3/p4_yosys.log
python3 - <<'PY'
import json
from collections import Counter

def count(path,top):
 d=json.load(open(path)); return Counter(c['type'] for c in d['modules'][top].get('cells',{}).values())
def show(name,c):
 lut=sum(v for k,v in c.items() if k.startswith('LUT'))
 ff=sum(v for k,v in c.items() if k.startswith('FD'))
 print(f"SYNTH {name}: DSP48E1={c.get('DSP48E1',0)} RAMB36E1={c.get('RAMB36E1',0)} RAMB18E1={c.get('RAMB18E1',0)} LUT={lut} FF={ff}")
 return lut,ff
c2=count('build3/conv2_engine.json','conv2_p3q4_pixel_engine')
p4=count('build3/p4_engine.json','conv_p4q2_pixel_engine')
show('CONV2_ENGINE',c2);show('P4Q2_ENGINE',p4)
if c2.get('DSP48E1',0)!=54: raise SystemExit(f"Conv2 engine DSP mismatch {c2.get('DSP48E1',0)}")
if p4.get('DSP48E1',0)!=36: raise SystemExit(f"P4Q2 engine DSP mismatch {p4.get('DSP48E1',0)}")
print('PIXEL_ENGINE_RESOURCE_ASSERTIONS_PASS')
PY

echo "STAGE3_ALL_PASS"
