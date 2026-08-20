`timescale 1ns/1ps
module tb_pixel_engines;
    logic clk=0, rst_n=0;
    always #5 clk=~clk;
    integer cycle=0;
    always @(posedge clk) cycle<=cycle+1;

    logic c2_valid,c2_first,c2_last,c2_out_valid;
    logic signed [215:0] c2_x,c2_w0,c2_w1,c2_w2,c2_w3;
    logic signed [31:0] c2_o0,c2_o1,c2_o2,c2_o3;
    conv2_p3q4_pixel_engine u_c2(
        .clk(clk),.rst_n(rst_n),.in_valid(c2_valid),.group_first(c2_first),.group_last(c2_last),
        .x_flat(c2_x),.w0_flat(c2_w0),.w1_flat(c2_w1),.w2_flat(c2_w2),.w3_flat(c2_w3),
        .out_valid(c2_out_valid),.out0(c2_o0),.out1(c2_o1),.out2(c2_o2),.out3(c2_o3)
    );

    logic p4_valid,p4_first,p4_last,p4_out_valid;
    logic signed [287:0] p4_x,p4_w0,p4_w1;
    logic signed [31:0] p4_o0,p4_o1;
    conv_p4q2_pixel_engine u_p4(
        .clk(clk),.rst_n(rst_n),.in_valid(p4_valid),.group_first(p4_first),.group_last(p4_last),
        .x_flat(p4_x),.w0_flat(p4_w0),.w1_flat(p4_w1),
        .out_valid(p4_out_valid),.out0(p4_o0),.out1(p4_o1)
    );

    integer errors=0;
    integer outputs_checked=0;
    integer pix,og,cg,l,tap,k,ic,oc;
    integer xv,wv0,wv1,wv2,wv3;
    integer e0,e1,e2,e3;
    integer cstart;

    function integer aval(input integer stage,input integer p,input integer ci,input integer kt);
        aval = ((stage*41 + p*23 + ci*17 + kt*29 + 11) & 255) - 128;
    endfunction
    function integer wval(input integer stage,input integer co,input integer ci,input integer kt);
        wval = ((stage*37 + co*31 + ci*13 + kt*7 + 19) & 255) - 128;
    endfunction

    task automatic drive_conv2_pixel(input integer p);
        integer local_start;
        begin
            local_start=cycle;
            for(og=0;og<8;og=og+1) begin
                e0=0;e1=0;e2=0;e3=0;
                for(cg=0;cg<11;cg=cg+1) begin
                    c2_x='0;c2_w0='0;c2_w1='0;c2_w2='0;c2_w3='0;
                    for(l=0;l<3;l=l+1) begin
                        ic=cg*3+l;
                        for(tap=0;tap<9;tap=tap+1) begin
                            k=l*9+tap;
                            if(ic<32) begin
                                xv=aval(2,p,ic,tap);
                                wv0=wval(2,og*4+0,ic,tap);
                                wv1=wval(2,og*4+1,ic,tap);
                                wv2=wval(2,og*4+2,ic,tap);
                                wv3=wval(2,og*4+3,ic,tap);
                            end else begin
                                xv=0;wv0=0;wv1=0;wv2=0;wv3=0;
                            end
                            c2_x [k*8 +:8]=xv;
                            c2_w0[k*8 +:8]=wv0;c2_w1[k*8 +:8]=wv1;
                            c2_w2[k*8 +:8]=wv2;c2_w3[k*8 +:8]=wv3;
                            e0=e0+xv*wv0;e1=e1+xv*wv1;e2=e2+xv*wv2;e3=e3+xv*wv3;
                        end
                    end
                    c2_first=(cg==0); c2_last=(cg==10); c2_valid=1'b1;
                    @(posedge clk); #1;
                    if(cg==10) begin
                        if(!c2_out_valid) begin $display("C2_MISSING_VALID p=%0d og=%0d",p,og); errors=errors+1; end
                        if($signed(c2_o0)!==e0) errors=errors+1;
                        if($signed(c2_o1)!==e1) errors=errors+1;
                        if($signed(c2_o2)!==e2) errors=errors+1;
                        if($signed(c2_o3)!==e3) errors=errors+1;
                        outputs_checked=outputs_checked+4;
                    end
                end
            end
            c2_valid=0;c2_first=0;c2_last=0;
            if((cycle-local_start)!=88) begin
                $display("C2_CYCLE_MISMATCH got=%0d exp=88",cycle-local_start);errors=errors+1;
            end
        end
    endtask

    task automatic drive_p4q2_pixel(input integer stage,input integer cin_count,input integer p);
        integer groups,local_start;
        begin
            groups=(cin_count+3)/4;
            local_start=cycle;
            for(og=0;og<32;og=og+1) begin
                e0=0;e1=0;
                for(cg=0;cg<groups;cg=cg+1) begin
                    p4_x='0;p4_w0='0;p4_w1='0;
                    for(l=0;l<4;l=l+1) begin
                        ic=cg*4+l;
                        for(tap=0;tap<9;tap=tap+1) begin
                            k=l*9+tap;
                            if(ic<cin_count) begin
                                xv=aval(stage,p,ic,tap);
                                wv0=wval(stage,og*2+0,ic,tap);
                                wv1=wval(stage,og*2+1,ic,tap);
                            end else begin xv=0;wv0=0;wv1=0; end
                            p4_x[k*8 +:8]=xv;p4_w0[k*8 +:8]=wv0;p4_w1[k*8 +:8]=wv1;
                            e0=e0+xv*wv0;e1=e1+xv*wv1;
                        end
                    end
                    p4_first=(cg==0);p4_last=(cg==(groups-1));p4_valid=1'b1;
                    @(posedge clk);#1;
                    if(cg==(groups-1)) begin
                        if(!p4_out_valid) begin $display("P4_MISSING_VALID stage=%0d p=%0d og=%0d",stage,p,og);errors=errors+1;end
                        if($signed(p4_o0)!==e0) errors=errors+1;
                        if($signed(p4_o1)!==e1) errors=errors+1;
                        outputs_checked=outputs_checked+2;
                    end
                end
            end
            p4_valid=0;p4_first=0;p4_last=0;
            if((cycle-local_start)!=(32*groups)) begin
                $display("P4_CYCLE_MISMATCH stage=%0d got=%0d exp=%0d",stage,cycle-local_start,32*groups);errors=errors+1;
            end
        end
    endtask

    initial begin
        c2_valid=0;c2_first=0;c2_last=0;c2_x='0;c2_w0='0;c2_w1='0;c2_w2='0;c2_w3='0;
        p4_valid=0;p4_first=0;p4_last=0;p4_x='0;p4_w0='0;p4_w1='0;
        repeat(5) @(posedge clk);rst_n<=1'b1;repeat(2) @(posedge clk);

        for(pix=0;pix<4;pix=pix+1) drive_conv2_pixel(pix);
        for(pix=0;pix<2;pix=pix+1) drive_p4q2_pixel(3,32,pix);
        drive_p4q2_pixel(4,64,0);

        $display("PIXEL_ENGINES outputs_checked=%0d errors=%0d",outputs_checked,errors);
        $display("PIXEL_SCHEDULE Conv2=88cycles Conv3=256cycles Conv4=512cycles per valid spatial pixel");
        if(errors!=0) $fatal(1,"pixel engine functional/cycle verification failed");
        $display("PIXEL_ENGINES_PASS");
        $finish;
    end
endmodule
