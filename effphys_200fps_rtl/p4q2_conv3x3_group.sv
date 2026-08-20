`timescale 1ns/1ps
module p4q2_conv3x3_group (
    input  logic signed [287:0] x_flat,
    input  logic signed [287:0] w0_flat,
    input  logic signed [287:0] w1_flat,
    output logic signed [23:0] sum0,
    output logic signed [23:0] sum1
);
    // 4 input channels x 3x3 kernel = 36 shared activations.
    // One packed multiplier per activation yields two output-channel products.
    logic signed [15:0] p0 [0:35];
    logic signed [15:0] p1 [0:35];

    genvar g;
    generate
        for (g=0; g<36; g=g+1) begin : G_P4Q2
            int8_packed_mul2 u01(
                .x(x_flat[g*8 +: 8]),
                .w0(w0_flat[g*8 +: 8]), .w1(w1_flat[g*8 +: 8]),
                .p0(p0[g]), .p1(p1[g])
            );
        end
    endgenerate

    integer i;
    always_comb begin
        sum0='0; sum1='0;
        for (i=0; i<36; i=i+1) begin
            sum0 = sum0 + p0[i];
            sum1 = sum1 + p1[i];
        end
    end
endmodule
