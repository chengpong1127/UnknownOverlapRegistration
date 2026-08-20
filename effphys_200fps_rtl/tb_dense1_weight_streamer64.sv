`timescale 1ns/1ps
module tb_dense1_weight_streamer64;
    localparam integer TOTAL_BYTES = 16384;
    localparam integer TOTAL_WORDS = TOTAL_BYTES/4;
    localparam integer N_BURSTS    = TOTAL_BYTES/128;
    localparam logic [31:0] BASE_ADDR = 32'h1000_0000;

    logic clk=0, rst_n=0, start=0;
    always #5 clk = ~clk;

    logic [31:0] araddr;
    logic [7:0]  arlen;
    logic [2:0]  arsize;
    logic [1:0]  arburst;
    logic        arvalid, arready;
    logic [63:0] rdata;
    logic        rvalid, rlast, rready;
    logic [31:0] weight_data;
    logic        weight_valid, weight_ready;
    logic        busy, done;
    logic [3:0]  dbg_outstanding;

    dense1_weight_streamer64 #(
        .TOTAL_BYTES(TOTAL_BYTES), .FIFO_DEPTH(256), .MAX_OUTSTANDING(8)
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .base_addr(BASE_ADDR),
        .m_axi_araddr(araddr), .m_axi_arlen(arlen), .m_axi_arsize(arsize),
        .m_axi_arburst(arburst), .m_axi_arvalid(arvalid), .m_axi_arready(arready),
        .m_axi_rdata(rdata), .m_axi_rvalid(rvalid), .m_axi_rlast(rlast), .m_axi_rready(rready),
        .weight_data(weight_data), .weight_valid(weight_valid), .weight_ready(weight_ready),
        .busy(busy), .done(done), .dbg_outstanding(dbg_outstanding)
    );

    logic [31:0] arq [0:255];
    integer qh=0, qt=0;
    integer beat=0;
    integer latency_count=12;
    logic resp_active=0;
    logic [31:0] current_addr=0;
    integer cycle=0;
    integer start_cycle=0;
    integer ar_count=0;
    integer expected_word=0;
    integer errors=0;
    integer test_outstanding=0;
    integer max_outstanding=0;
    integer starve_cycles=0;
    logic seen_first_word=0;

    always_comb begin
        rvalid = resp_active;
        rlast  = resp_active && (beat == 15);
        rdata[31:0]  = ((current_addr - BASE_ADDR) >> 2) + beat*2;
        rdata[63:32] = ((current_addr - BASE_ADDR) >> 2) + beat*2 + 1;
    end

    always @(posedge clk) begin
        cycle <= cycle + 1;
        arready <= ((cycle % 7) != 2); // deterministic command backpressure
        weight_ready <= 1'b1;

        if (arvalid && arready) begin
            if (arlen !== 8'd15 || arsize !== 3'd3 || arburst !== 2'b01) begin
                $display("AR_PROTOCOL_MISMATCH len=%0d size=%0d burst=%0d", arlen, arsize, arburst);
                errors <= errors + 1;
            end
            if (araddr !== (BASE_ADDR + ar_count*128)) begin
                $display("AR_ADDR_MISMATCH idx=%0d got=%h exp=%h", ar_count, araddr, BASE_ADDR + ar_count*128);
                errors <= errors + 1;
            end
            arq[qt] <= araddr;
            qt <= qt + 1;
            ar_count <= ar_count + 1;
        end

        case ({(arvalid && arready), (rvalid && rready && rlast)})
            2'b10: begin
                test_outstanding <= test_outstanding + 1;
                if ((test_outstanding + 1) > max_outstanding)
                    max_outstanding <= test_outstanding + 1;
            end
            2'b01: test_outstanding <= test_outstanding - 1;
            default: test_outstanding <= test_outstanding;
        endcase

        if (!resp_active) begin
            if (qh < qt) begin
                if (latency_count > 0) begin
                    latency_count <= latency_count - 1;
                end else begin
                    current_addr <= arq[qh];
                    qh <= qh + 1;
                    beat <= 0;
                    resp_active <= 1'b1;
                end
            end
        end else if (rvalid && rready) begin
            if (beat == 15) begin
                if (qh < qt) begin
                    current_addr <= arq[qh];
                    qh <= qh + 1;
                    beat <= 0;
                    resp_active <= 1'b1; // queued bursts return back-to-back
                end else begin
                    resp_active <= 1'b0;
                    latency_count <= 12;
                    beat <= 0;
                end
            end else begin
                beat <= beat + 1;
            end
        end

        if (weight_valid && weight_ready) begin
            seen_first_word <= 1'b1;
            if (weight_data !== expected_word[31:0]) begin
                if (errors < 8)
                    $display("WEIGHT_MISMATCH word=%0d got=%h exp=%h", expected_word, weight_data, expected_word[31:0]);
                errors <= errors + 1;
            end
            expected_word <= expected_word + 1;
        end else if (seen_first_word && busy && weight_ready && !weight_valid) begin
            starve_cycles <= starve_cycles + 1;
        end
    end

    initial begin
        arready=0; weight_ready=0;
        repeat(5) @(posedge clk);
        rst_n <= 1'b1;
        repeat(2) @(posedge clk);
        start_cycle = cycle;
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;
        wait(done === 1'b1);
        #1;
        $display("DENSE64 words=%0d bursts=%0d cycles=%0d max_outstanding=%0d starve_cycles=%0d errors=%0d",
                 expected_word, ar_count, cycle-start_cycle, max_outstanding, starve_cycles, errors);
        if (expected_word != TOTAL_WORDS) $fatal(1,"wrong output word count");
        if (ar_count != N_BURSTS) $fatal(1,"wrong AXI burst count");
        if (max_outstanding < 4) $fatal(1,"multiple outstanding reads were not achieved");
        if (starve_cycles != 0) $fatal(1,"weight stream starved after startup");
        if ((cycle-start_cycle) > (TOTAL_WORDS+100)) $fatal(1,"streamer failed one-word-per-cycle target");
        if (errors != 0) $fatal(1,"stream data/protocol mismatches detected");
        $display("DENSE64_PASS");
        $finish;
    end

    initial begin
        #2000000;
        $fatal(1,"timeout");
    end
endmodule
