`timescale 1ns/1ps
module tb_winograd_f2x2_3x3_int8;
    reg [16*8-1:0] tile_flat;
    reg [9*8-1:0] kern_flat;
    wire signed [31:0] y00, y01, y10, y11;

    integer d [0:15];
    integer g [0:8];
    integer exp00, exp01, exp10, exp11;
    integer t, idx;
    integer seed;
    integer rv;

    winograd_f2x2_3x3_int8 dut (
        .tile_flat(tile_flat), .kern_flat(kern_flat),
        .y00(y00), .y01(y01), .y10(y10), .y11(y11)
    );

    task pack_inputs;
        integer i;
        begin
            for (i=0; i<16; i=i+1)
                tile_flat[i*8 +: 8] = d[i][7:0];
            for (i=0; i<9; i=i+1)
                kern_flat[i*8 +: 8] = g[i][7:0];
        end
    endtask

    task calc_direct;
        integer rr, cc;
        begin
            exp00 = 0; exp01 = 0; exp10 = 0; exp11 = 0;
            for (rr=0; rr<3; rr=rr+1) begin
                for (cc=0; cc<3; cc=cc+1) begin
                    exp00 = exp00 + g[rr*3+cc] * d[(rr+0)*4+(cc+0)];
                    exp01 = exp01 + g[rr*3+cc] * d[(rr+0)*4+(cc+1)];
                    exp10 = exp10 + g[rr*3+cc] * d[(rr+1)*4+(cc+0)];
                    exp11 = exp11 + g[rr*3+cc] * d[(rr+1)*4+(cc+1)];
                end
            end
        end
    endtask

    task check_case;
        input integer case_id;
        begin
            pack_inputs();
            calc_direct();
            #1;
            if (($signed(y00) !== exp00) || ($signed(y01) !== exp01) ||
                ($signed(y10) !== exp10) || ($signed(y11) !== exp11)) begin
                $display("FAIL case=%0d", case_id);
                $display("got %0d %0d %0d %0d", $signed(y00), $signed(y01), $signed(y10), $signed(y11));
                $display("exp %0d %0d %0d %0d", exp00, exp01, exp10, exp11);
                $fatal(1);
            end
        end
    endtask

    initial begin
        seed = 32'h35A91C72;
        tile_flat = 0; kern_flat = 0;

        for (idx=0; idx<16; idx=idx+1) d[idx] = 127;
        for (idx=0; idx<9; idx=idx+1) g[idx] = 127;
        check_case(-1);
        for (idx=0; idx<16; idx=idx+1) d[idx] = -128;
        for (idx=0; idx<9; idx=idx+1) g[idx] = -128;
        check_case(-2);
        for (idx=0; idx<16; idx=idx+1) d[idx] = (idx[0] ? -128 : 127);
        for (idx=0; idx<9; idx=idx+1) g[idx] = (idx[0] ? 127 : -128);
        check_case(-3);

        for (t=0; t<5000; t=t+1) begin
            for (idx=0; idx<16; idx=idx+1) begin
                rv = $random(seed);
                d[idx] = (rv & 8'hff);
                if (d[idx] >= 128) d[idx] = d[idx] - 256;
            end
            for (idx=0; idx<9; idx=idx+1) begin
                rv = $random(seed);
                g[idx] = (rv & 8'hff);
                if (g[idx] >= 128) g[idx] = g[idx] - 256;
            end
            check_case(t);
        end
        $display("PASS: 5003 exact INT8 F(2x2,3x3) cases");
        $finish;
    end
endmodule
