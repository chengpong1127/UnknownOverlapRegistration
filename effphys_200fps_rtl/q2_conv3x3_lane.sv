`timescale 1ns/1ps
module q2_conv3x3_lane (
    input  logic signed [71:0] x_flat,
    input  logic signed [71:0] w0_flat,
    input  logic signed [71:0] w1_flat,
    output logic signed [23:0] sum0,
    output logic signed [23:0] sum1
);
    logic signed [15:0] p0 [0:8];
    logic signed [15:0] p1 [0:8];

    genvar g;
    generate
        for (g = 0; g < 9; g = g + 1) begin : G_PACKED_MAC
            int8_packed_mul2 u_mul2 (
                .x  (x_flat [g*8 +: 8]),
                .w0 (w0_flat[g*8 +: 8]),
                .w1 (w1_flat[g*8 +: 8]),
                .p0 (p0[g]),
                .p1 (p1[g])
            );
        end
    endgenerate

    integer i;
    always_comb begin
        sum0 = '0;
        sum1 = '0;
        for (i = 0; i < 9; i = i + 1) begin
            sum0 = sum0 + p0[i];
            sum1 = sum1 + p1[i];
        end
    end
endmodule
