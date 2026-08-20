`timescale 1ns/1ps
module tb_int8_packed_mul2;
    logic signed [7:0] x, w0, w1;
    logic signed [15:0] p0, p1;
    integer xi, wi0, wi1;
    integer errors;
    integer checks;
    integer exp0, exp1;

    int8_packed_mul2 dut(.x(x), .w0(w0), .w1(w1), .p0(p0), .p1(p1));

    initial begin
        x = '0; w0 = '0; w1 = '0;
        errors = 0;
        checks = 0;
        for (xi = -128; xi <= 127; xi = xi + 1) begin
            for (wi0 = -128; wi0 <= 127; wi0 = wi0 + 1) begin
                for (wi1 = -128; wi1 <= 127; wi1 = wi1 + 1) begin
                    x  = xi;
                    w0 = wi0;
                    w1 = wi1;
                    #1;
                    exp0 = xi * wi0;
                    exp1 = xi * wi1;
                    checks = checks + 2;
                    if ($signed(p0) !== exp0) begin
                        if (errors < 8)
                            $display("P0_MISMATCH x=%0d w=%0d got=%0d exp=%0d", xi, wi0, $signed(p0), exp0);
                        errors = errors + 1;
                    end
                    if ($signed(p1) !== exp1) begin
                        if (errors < 8)
                            $display("P1_MISMATCH x=%0d w=%0d got=%0d exp=%0d", xi, wi1, $signed(p1), exp1);
                        errors = errors + 1;
                    end
                end
            end
        end
        $display("PACK_EXHAUSTIVE triples=%0d products=%0d errors=%0d", 256*256*256, checks, errors);
        if (errors != 0) $fatal(1, "packed INT8 multiplier failed exhaustive verification");
        $display("PACK_EXHAUSTIVE_PASS");
        $finish;
    end
endmodule
