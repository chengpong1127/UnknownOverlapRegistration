`timescale 1ns/1ps
module p3q4_conv3x3_group (
    input  logic signed [215:0] x_flat,
    input  logic signed [215:0] w0_flat,
    input  logic signed [215:0] w1_flat,
    input  logic signed [215:0] w2_flat,
    input  logic signed [215:0] w3_flat,
    output logic signed [23:0] sum0,
    output logic signed [23:0] sum1,
    output logic signed [23:0] sum2,
    output logic signed [23:0] sum3
);
    // 3 input channels x 3x3 kernel = 27 shared activations.
    // Each activation uses two packed multipliers, producing four output-channel products.
    logic signed [15:0] p0 [0:26];
    logic signed [15:0] p1 [0:26];
    logic signed [15:0] p2 [0:26];
    logic signed [15:0] p3 [0:26];

    genvar g;
    generate
        for (g=0; g<27; g=g+1) begin : G_P3Q4
            int8_packed_mul2 u01(
                .x(x_flat[g*8 +: 8]),
                .w0(w0_flat[g*8 +: 8]), .w1(w1_flat[g*8 +: 8]),
                .p0(p0[g]), .p1(p1[g])
            );
            int8_packed_mul2 u23(
                .x(x_flat[g*8 +: 8]),
                .w0(w2_flat[g*8 +: 8]), .w1(w3_flat[g*8 +: 8]),
                .p0(p2[g]), .p1(p3[g])
            );
        end
    endgenerate

    integer i;
    always_comb begin
        sum0='0; sum1='0; sum2='0; sum3='0;
        for (i=0; i<27; i=i+1) begin
            sum0 = sum0 + p0[i];
            sum1 = sum1 + p1[i];
            sum2 = sum2 + p2[i];
            sum3 = sum3 + p3[i];
        end
    end
endmodule
