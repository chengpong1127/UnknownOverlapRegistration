`timescale 1ns/1ps
module tb_q2_conv3x3_lane;
    logic signed [71:0] x_flat, w0_flat, w1_flat;
    logic signed [23:0] sum0, sum1;
    integer n, k;
    integer xv, a, b;
    integer exp0, exp1;
    integer errors;

    q2_conv3x3_lane dut(
        .x_flat(x_flat), .w0_flat(w0_flat), .w1_flat(w1_flat),
        .sum0(sum0), .sum1(sum1)
    );

    initial begin
        x_flat='0; w0_flat='0; w1_flat='0; errors=0;
        for (n=0; n<20000; n=n+1) begin
            exp0=0; exp1=0;
            for (k=0; k<9; k=k+1) begin
                xv = ((n*17 + k*29 + 11) & 255) - 128;
                a  = ((n*31 + k*13 + 73) & 255) - 128;
                b  = ((n*43 + k*7  + 19) & 255) - 128;
                x_flat [k*8 +: 8] = xv;
                w0_flat[k*8 +: 8] = a;
                w1_flat[k*8 +: 8] = b;
                exp0 = exp0 + xv*a;
                exp1 = exp1 + xv*b;
            end
            #1;
            if ($signed(sum0) !== exp0) begin
                if (errors < 8) $display("Q2_SUM0_MISMATCH n=%0d got=%0d exp=%0d",n,$signed(sum0),exp0);
                errors=errors+1;
            end
            if ($signed(sum1) !== exp1) begin
                if (errors < 8) $display("Q2_SUM1_MISMATCH n=%0d got=%0d exp=%0d",n,$signed(sum1),exp1);
                errors=errors+1;
            end
        end
        $display("Q2_LANE vectors=20000 outputs_checked=40000 errors=%0d",errors);
        if (errors != 0) $fatal(1,"Q2 lane verification failed");
        $display("Q2_LANE_PASS");
        $finish;
    end
endmodule
