`timescale 1ns/1ps
module tb_pq_groups;
    logic signed [215:0] x3,w30,w31,w32,w33;
    logic signed [23:0] s30,s31,s32,s33;
    logic signed [287:0] x4,w40,w41;
    logic signed [23:0] s40,s41;
    integer n,k,xv,a,b,c,d;
    integer e30,e31,e32,e33,e40,e41;
    integer errors;

    p3q4_conv3x3_group u_p3q4(
        .x_flat(x3),.w0_flat(w30),.w1_flat(w31),.w2_flat(w32),.w3_flat(w33),
        .sum0(s30),.sum1(s31),.sum2(s32),.sum3(s33)
    );
    p4q2_conv3x3_group u_p4q2(
        .x_flat(x4),.w0_flat(w40),.w1_flat(w41),.sum0(s40),.sum1(s41)
    );

    initial begin
        x3='0;w30='0;w31='0;w32='0;w33='0;
        x4='0;w40='0;w41='0;errors=0;
        for(n=0;n<5000;n=n+1) begin
            e30=0;e31=0;e32=0;e33=0;
            for(k=0;k<27;k=k+1) begin
                xv=((n*17+k*29+3)&255)-128;
                a =((n*31+k*11+7)&255)-128;
                b =((n*43+k*13+9)&255)-128;
                c =((n*19+k*23+5)&255)-128;
                d =((n*47+k*5 +1)&255)-128;
                // Exercise the Conv2 tail group: every 17th vector zeroes the
                // third input-channel 3x3 block, equivalent to the final Cin=32 P=3 tail.
                if((n%17)==0 && k>=18) begin xv=0; a=0; b=0; c=0; d=0; end
                x3 [k*8 +:8]=xv; w30[k*8 +:8]=a; w31[k*8 +:8]=b;
                w32[k*8 +:8]=c; w33[k*8 +:8]=d;
                e30=e30+xv*a; e31=e31+xv*b; e32=e32+xv*c; e33=e33+xv*d;
            end
            e40=0;e41=0;
            for(k=0;k<36;k=k+1) begin
                xv=((n*37+k*17+15)&255)-128;
                a =((n*13+k*31+27)&255)-128;
                b =((n*53+k*7 +39)&255)-128;
                x4[k*8 +:8]=xv; w40[k*8 +:8]=a; w41[k*8 +:8]=b;
                e40=e40+xv*a; e41=e41+xv*b;
            end
            #1;
            if($signed(s30)!==e30) errors=errors+1;
            if($signed(s31)!==e31) errors=errors+1;
            if($signed(s32)!==e32) errors=errors+1;
            if($signed(s33)!==e33) errors=errors+1;
            if($signed(s40)!==e40) errors=errors+1;
            if($signed(s41)!==e41) errors=errors+1;
            if(errors!=0 && errors<8)
                $display("PQ_MISMATCH n=%0d p3q4=(%0d,%0d,%0d,%0d)/(%0d,%0d,%0d,%0d) p4q2=(%0d,%0d)/(%0d,%0d)",
                    n,$signed(s30),$signed(s31),$signed(s32),$signed(s33),e30,e31,e32,e33,
                    $signed(s40),$signed(s41),e40,e41);
        end
        $display("PQ_GROUPS vectors=5000 scalar_output_sums_checked=30000 errors=%0d",errors);
        if(errors!=0) $fatal(1,"packed P/Q group verification failed");
        $display("PQ_GROUPS_PASS");
        $finish;
    end
endmodule
