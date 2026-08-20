`timescale 1ns/1ps
module int8_packed_mul2 (
    input  logic signed [7:0]  x,
    input  logic signed [7:0]  w0,
    input  logic signed [7:0]  w1,
    output logic signed [15:0] p0,
    output logic signed [15:0] p1
);
    // Offset signed INT8 to [0,255].  The high bit of these 9-bit values
    // is always zero, so the packed multiplier width remains 24 x 9.
    logic [8:0] xu, w0u, w1u;
    assign xu  = $signed(x)  + 9'sd128;
    assign w0u = $signed(w0) + 9'sd128;
    assign w1u = $signed(w1) + 9'sd128;

    // v0 occupies bits 7:0 and v1 occupies bits 23:16.  Since
    // xu*v0 < 2^16, the two unsigned products do not overlap.
    logic [23:0] packed_w;
    (* use_dsp = "yes" *) logic [32:0] packed_p;
    assign packed_w = {w1u[7:0], 8'h00, w0u[7:0]};
    assign packed_p = packed_w * xu;

    logic [15:0] raw0, raw1;
    assign raw0 = packed_p[15:0];
    assign raw1 = packed_p[31:16];

    // x*w = (x+128)(w+128) - 128(x+128) - 128(w+128) + 16384.
    // All correction terms are shifts/adds; only packed_p is a multiplier.
    logic signed [17:0] corr0, corr1;
    assign corr0 = $signed({2'b00, raw0})
                 - $signed({2'b00, xu,  7'b0})
                 - $signed({2'b00, w0u, 7'b0})
                 + 18'sd16384;
    assign corr1 = $signed({2'b00, raw1})
                 - $signed({2'b00, xu,  7'b0})
                 - $signed({2'b00, w1u, 7'b0})
                 + 18'sd16384;

    assign p0 = corr0[15:0];
    assign p1 = corr1[15:0];
endmodule
