`timescale 1ns/1ps
module conv2_p3q4_pixel_engine (
    input  logic clk,
    input  logic rst_n,
    input  logic in_valid,
    input  logic group_first,
    input  logic group_last,
    input  logic signed [215:0] x_flat,
    input  logic signed [215:0] w0_flat,
    input  logic signed [215:0] w1_flat,
    input  logic signed [215:0] w2_flat,
    input  logic signed [215:0] w3_flat,
    output logic out_valid,
    output logic signed [31:0] out0,
    output logic signed [31:0] out1,
    output logic signed [31:0] out2,
    output logic signed [31:0] out3
);
    logic signed [23:0] g0,g1,g2,g3;
    logic signed [31:0] a0,a1,a2,a3;

    p3q4_conv3x3_group u_group(
        .x_flat(x_flat),.w0_flat(w0_flat),.w1_flat(w1_flat),
        .w2_flat(w2_flat),.w3_flat(w3_flat),
        .sum0(g0),.sum1(g1),.sum2(g2),.sum3(g3)
    );

    always_ff @(posedge clk) begin
        if(!rst_n) begin
            a0<='0;a1<='0;a2<='0;a3<='0;
            out0<='0;out1<='0;out2<='0;out3<='0;
            out_valid<=1'b0;
        end else begin
            out_valid<=1'b0;
            if(in_valid) begin
                if(group_first) begin
                    a0<=g0; a1<=g1; a2<=g2; a3<=g3;
                end else begin
                    a0<=a0+g0; a1<=a1+g1; a2<=a2+g2; a3<=a3+g3;
                end
                if(group_last) begin
                    out0 <= group_first ? $signed(g0) : a0+$signed(g0);
                    out1 <= group_first ? $signed(g1) : a1+$signed(g1);
                    out2 <= group_first ? $signed(g2) : a2+$signed(g2);
                    out3 <= group_first ? $signed(g3) : a3+$signed(g3);
                    out_valid<=1'b1;
                end
            end
        end
    end
endmodule
