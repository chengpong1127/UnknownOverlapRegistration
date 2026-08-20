`timescale 1ns/1ps

// Exact integer Winograd F(2x2,3x3) primitive for signed INT8 data/weights.
// Uses G2 = 2*G so all transforms are integer. The inverse output is divided by 4.
// This module is intentionally wide/unpipelined as a correctness prototype.
module winograd_f2x2_3x3_int8 (
    input  wire [16*8-1:0] tile_flat,
    input  wire [ 9*8-1:0] kern_flat,
    output wire signed [31:0] y00,
    output wire signed [31:0] y01,
    output wire signed [31:0] y10,
    output wire signed [31:0] y11
);
    wire signed [7:0] d [0:15];
    wire signed [7:0] g [0:8];

    genvar gi;
    generate
        for (gi = 0; gi < 16; gi = gi + 1) begin : GEN_D
            assign d[gi] = $signed(tile_flat[gi*8 +: 8]);
        end
        for (gi = 0; gi < 9; gi = gi + 1) begin : GEN_G
            assign g[gi] = $signed(kern_flat[gi*8 +: 8]);
        end
    endgenerate

    wire signed [31:0] tr [0:15];
    wire signed [31:0] v  [0:15];

    generate
        for (gi = 0; gi < 4; gi = gi + 1) begin : GEN_D_ROW
            assign tr[4*gi+0] = $signed(d[4*gi+0]) - $signed(d[4*gi+2]);
            assign tr[4*gi+1] = $signed(d[4*gi+1]) + $signed(d[4*gi+2]);
            assign tr[4*gi+2] = -$signed(d[4*gi+1]) + $signed(d[4*gi+2]);
            assign tr[4*gi+3] = $signed(d[4*gi+1]) - $signed(d[4*gi+3]);
        end
    endgenerate

    generate
        for (gi = 0; gi < 4; gi = gi + 1) begin : GEN_D_COL
            assign v[0*4+gi] = tr[0*4+gi] - tr[2*4+gi];
            assign v[1*4+gi] = tr[1*4+gi] + tr[2*4+gi];
            assign v[2*4+gi] = -tr[1*4+gi] + tr[2*4+gi];
            assign v[3*4+gi] = tr[1*4+gi] - tr[3*4+gi];
        end
    endgenerate

    wire signed [31:0] gr [0:11];
    wire signed [31:0] u  [0:15];

    generate
        for (gi = 0; gi < 3; gi = gi + 1) begin : GEN_G_ROW
            assign gr[0*3+gi] = 2 * $signed(g[0*3+gi]);
            assign gr[1*3+gi] = $signed(g[0*3+gi]) + $signed(g[1*3+gi]) + $signed(g[2*3+gi]);
            assign gr[2*3+gi] = $signed(g[0*3+gi]) - $signed(g[1*3+gi]) + $signed(g[2*3+gi]);
            assign gr[3*3+gi] = 2 * $signed(g[2*3+gi]);
        end
        for (gi = 0; gi < 4; gi = gi + 1) begin : GEN_G_COL
            assign u[gi*4+0] = 2 * gr[gi*3+0];
            assign u[gi*4+1] = gr[gi*3+0] + gr[gi*3+1] + gr[gi*3+2];
            assign u[gi*4+2] = gr[gi*3+0] - gr[gi*3+1] + gr[gi*3+2];
            assign u[gi*4+3] = 2 * gr[gi*3+2];
        end
    endgenerate

    wire signed [63:0] m [0:15];
    generate
        for (gi = 0; gi < 16; gi = gi + 1) begin : GEN_M
            assign m[gi] = $signed(u[gi]) * $signed(v[gi]);
        end
    endgenerate

    wire signed [63:0] sr0 [0:3];
    wire signed [63:0] sr1 [0:3];
    generate
        for (gi = 0; gi < 4; gi = gi + 1) begin : GEN_Y_ROW
            assign sr0[gi] = m[0*4+gi] + m[1*4+gi] + m[2*4+gi];
            assign sr1[gi] = m[1*4+gi] - m[2*4+gi] - m[3*4+gi];
        end
    endgenerate

    wire signed [63:0] y00_x4 = sr0[0] + sr0[1] + sr0[2];
    wire signed [63:0] y01_x4 = sr0[1] - sr0[2] - sr0[3];
    wire signed [63:0] y10_x4 = sr1[0] + sr1[1] + sr1[2];
    wire signed [63:0] y11_x4 = sr1[1] - sr1[2] - sr1[3];

    assign y00 = $signed(y00_x4 >>> 2);
    assign y01 = $signed(y01_x4 >>> 2);
    assign y10 = $signed(y10_x4 >>> 2);
    assign y11 = $signed(y11_x4 >>> 2);
endmodule
