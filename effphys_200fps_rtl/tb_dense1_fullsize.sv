`timescale 1ns/1ps
module tb_dense1_fullsize;
    localparam integer TOTAL_BYTES = 2097152;
    localparam integer TOTAL_WORDS = TOTAL_BYTES/4;
    localparam integer N_BURSTS    = TOTAL_BYTES/128;
    localparam logic [31:0] BASE_ADDR = 32'h2000_0000;

    logic clk=0, rst_n=0, start=0;
    always #5 clk=~clk;

    logic [31:0] araddr;
    logic [7:0] arlen;
    logic [2:0] arsize;
    logic [1:0] arburst;
    logic arvalid,arready;
    logic [63:0] rdata;
    logic rvalid,rlast,rready;
    logic [31:0] weight_data;
    logic weight_valid,weight_ready,busy,done;
    logic [3:0] dbg_outstanding;

    dense1_weight_streamer64 #(
        .TOTAL_BYTES(TOTAL_BYTES),.FIFO_DEPTH(256),.MAX_OUTSTANDING(8)
    ) dut(
        .clk(clk),.rst_n(rst_n),.start(start),.base_addr(BASE_ADDR),
        .m_axi_araddr(araddr),.m_axi_arlen(arlen),.m_axi_arsize(arsize),.m_axi_arburst(arburst),
        .m_axi_arvalid(arvalid),.m_axi_arready(arready),
        .m_axi_rdata(rdata),.m_axi_rvalid(rvalid),.m_axi_rlast(rlast),.m_axi_rready(rready),
        .weight_data(weight_data),.weight_valid(weight_valid),.weight_ready(weight_ready),
        .busy(busy),.done(done),.dbg_outstanding(dbg_outstanding)
    );

    logic [31:0] arq [0:19999];
    integer qh=0,qt=0,beat=0,latency_count=12;
    logic resp_active=0;
    logic [31:0] current_addr=0;
    integer cycle=0,start_cycle=0,ar_count=0,expected_word=0,errors=0;
    integer test_outstanding=0,max_outstanding=0,starve_cycles=0;
    logic seen_first_word=0;

    always_comb begin
        // 25% RVALID bubbles stress the FIFO while keeping average 64-bit
        // supply bandwidth above the 32-bit/cycle Dense1 consumption rate.
        rvalid = resp_active && ((cycle % 4) != 1);
        rlast  = resp_active && (beat==15);
        rdata[31:0]  = ((current_addr-BASE_ADDR)>>2) + beat*2;
        rdata[63:32] = ((current_addr-BASE_ADDR)>>2) + beat*2 + 1;
    end

    always @(posedge clk) begin
        cycle<=cycle+1;
        arready<=((cycle%7)!=2);
        weight_ready<=1'b1;

        if(arvalid&&arready) begin
            if(arlen!==8'd15 || arsize!==3'd3 || arburst!==2'b01) errors<=errors+1;
            if(araddr!==(BASE_ADDR+ar_count*128)) errors<=errors+1;
            arq[qt]<=araddr; qt<=qt+1; ar_count<=ar_count+1;
        end

        case({(arvalid&&arready),(rvalid&&rready&&rlast)})
            2'b10: begin test_outstanding<=test_outstanding+1; if(test_outstanding+1>max_outstanding) max_outstanding<=test_outstanding+1; end
            2'b01: test_outstanding<=test_outstanding-1;
            default: test_outstanding<=test_outstanding;
        endcase

        if(!resp_active) begin
            if(qh<qt) begin
                if(latency_count>0) latency_count<=latency_count-1;
                else begin current_addr<=arq[qh];qh<=qh+1;beat<=0;resp_active<=1'b1;end
            end
        end else if(rvalid&&rready) begin
            if(beat==15) begin
                if(qh<qt) begin current_addr<=arq[qh];qh<=qh+1;beat<=0;resp_active<=1'b1;end
                else begin resp_active<=1'b0;latency_count<=12;beat<=0;end
            end else beat<=beat+1;
        end

        if(weight_valid&&weight_ready) begin
            seen_first_word<=1'b1;
            if(weight_data!==expected_word[31:0]) begin
                if(errors<8) $display("FULL_DENSE_MISMATCH idx=%0d got=%h exp=%h",expected_word,weight_data,expected_word[31:0]);
                errors<=errors+1;
            end
            expected_word<=expected_word+1;
        end else if(seen_first_word&&busy&&weight_ready&&!weight_valid) begin
            starve_cycles<=starve_cycles+1;
        end
    end

    initial begin
        arready=0;weight_ready=0;
        repeat(5) @(posedge clk);rst_n<=1'b1;repeat(2) @(posedge clk);
        start_cycle=cycle;start<=1'b1;@(posedge clk);start<=1'b0;
        wait(done===1'b1);#1;
        $display("DENSE64_FULL bytes=%0d words=%0d bursts=%0d cycles=%0d max_outstanding=%0d starve_cycles=%0d errors=%0d",
            TOTAL_BYTES,expected_word,ar_count,cycle-start_cycle,max_outstanding,starve_cycles,errors);
        if(expected_word!=TOTAL_WORDS) $fatal(1,"wrong full Dense1 output count");
        if(ar_count!=N_BURSTS) $fatal(1,"wrong full Dense1 burst count");
        if(max_outstanding!=8) $fatal(1,"did not reach 8 outstanding reads");
        if(starve_cycles!=0) $fatal(1,"full Dense1 stream starved after startup");
        if(errors!=0) $fatal(1,"full Dense1 data/protocol errors");
        if((cycle-start_cycle)>(TOTAL_WORDS+300)) $fatal(1,"full Dense1 stream exceeded near-1-word/cycle budget");
        $display("DENSE64_FULL_PASS");
        $finish;
    end

    initial begin
        #100000000;
        $fatal(1,"timeout");
    end
endmodule
