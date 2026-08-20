#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build2
exec > >(tee build2/STAGE2_RESULT.txt) 2>&1

echo "=== P3Q4 / P4Q2 FUNCTIONAL RTL SIM ==="
iverilog -g2012 -Wall -s tb_pq_groups -o build2/tb_pq.vvp \
  int8_packed_mul2.sv p3q4_conv3x3_group.sv p4q2_conv3x3_group.sv tb_pq_groups.sv
vvp build2/tb_pq.vvp

echo "=== VERILATOR LINT ==="
verilator --lint-only --timing -Wall -Wno-fatal \
  int8_packed_mul2.sv p3q4_conv3x3_group.sv p4q2_conv3x3_group.sv

echo "=== XC7 SYNTHESIS ==="
yosys -q -p 'read_verilog -sv int8_packed_mul2.sv p3q4_conv3x3_group.sv; synth_xilinx -family xc7 -noiopad -top p3q4_conv3x3_group; write_json build2/p3q4.json; stat' 2>&1 | tee build2/p3q4_yosys.log
yosys -q -p 'read_verilog -sv int8_packed_mul2.sv p4q2_conv3x3_group.sv; synth_xilinx -family xc7 -noiopad -top p4q2_conv3x3_group; write_json build2/p4q2.json; stat' 2>&1 | tee build2/p4q2_yosys.log

python3 - <<'PY'
import json, math
from collections import Counter

def cell_count(path,top):
    d=json.load(open(path))
    return Counter(c['type'] for c in d['modules'][top].get('cells',{}).values())

def summary(name,c):
    lut=sum(v for k,v in c.items() if k.startswith('LUT'))
    ff=sum(v for k,v in c.items() if k.startswith('FD'))
    print(f"SYNTH {name}: DSP48E1={c.get('DSP48E1',0)} RAMB36E1={c.get('RAMB36E1',0)} RAMB18E1={c.get('RAMB18E1',0)} LUT={lut} FF={ff}")

p3=cell_count('build2/p3q4.json','p3q4_conv3x3_group')
p4=cell_count('build2/p4q2.json','p4q2_conv3x3_group')
summary('P3Q4',p3); summary('P4Q2',p4)
if p3.get('DSP48E1',0)!=54: raise SystemExit(f"P3Q4 DSP mismatch: {p3.get('DSP48E1',0)} != 54")
if p4.get('DSP48E1',0)!=36: raise SystemExit(f"P4Q2 DSP mismatch: {p4.get('DSP48E1',0)} != 36")

def cyc(H,W,Cin,Cout,P,Q,pad):
    pre=2*H*W
    spatial=H*W if pad else (H-2)*(W-2)
    return pre + spatial*math.ceil(Cin/P)*math.ceil(Cout/Q)

cfg=[
 ('Conv1',72,72,3,32,1,1,True,9),
 ('Conv2',72,72,32,32,3,4,False,54),
 ('Conv3',35,35,32,64,4,2,True,36),
 ('Conv4',35,35,64,64,4,2,False,36),
]
rows=[]
for name,H,W,Cin,Cout,P,Q,pad,dsp in cfg:
    c=cyc(H,W,Cin,Cout,P,Q,pad)
    rows.append((name,c,dsp))
    print(f"CYCLE {name}: cycles={c} fps100={100e6/c:.6f} fps115={115e6/c:.6f} fps125={125e6/c:.6f} productDSP={dsp}")
b=max(c for _,c,_ in rows)
print(f"PIPELINE bottleneck_cycles={b} fmin_200={b*200/1e6:.6f}MHz fps115={115e6/b:.6f} fps125={125e6/b:.6f}")
print(f"CONV_PRODUCT_DSP total={sum(d for _,_,d in rows)}")
print(f"ROUGH_FULL_DSP current182-currentConv99+newConv135={182-99+135}")
if 115e6/b < 200: raise SystemExit('115 MHz does not meet 200 FPS cycle budget')
print('STAGE2_200FPS_BUDGET_PASS')
PY

echo "STAGE2_ALL_PASS"
