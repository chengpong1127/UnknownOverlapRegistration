`timescale 1ns/1ps
module dense1_weight_streamer64 #(
    parameter integer TOTAL_BYTES      = 2097152,
    parameter integer FIFO_DEPTH       = 256,
    parameter integer MAX_OUTSTANDING  = 8
) (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         start,
    input  logic [31:0]  base_addr,

    output logic [31:0]  m_axi_araddr,
    output logic [7:0]   m_axi_arlen,
    output logic [2:0]   m_axi_arsize,
    output logic [1:0]   m_axi_arburst,
    output logic         m_axi_arvalid,
    input  logic         m_axi_arready,

    input  logic [63:0]  m_axi_rdata,
    input  logic         m_axi_rvalid,
    input  logic         m_axi_rlast,
    output logic         m_axi_rready,

    output logic [31:0]  weight_data,
    output logic         weight_valid,
    input  logic         weight_ready,

    output logic         busy,
    output logic         done,
    output logic [$clog2(MAX_OUTSTANDING+1)-1:0] dbg_outstanding
);
    localparam integer PTR_W       = $clog2(FIFO_DEPTH);
    localparam integer OUT_W       = $clog2(MAX_OUTSTANDING+1);
    localparam integer TOTAL_WORDS = TOTAL_BYTES/4;
    localparam integer BURST_BYTES = 16*8;

    (* ram_style = "block" *) logic [63:0] fifo_mem [0:FIFO_DEPTH-1];
    logic [PTR_W-1:0] wr_ptr, rd_ptr;
    logic [PTR_W:0]   fifo_count;
    logic             half_sel;

    logic [31:0] base_addr_q;
    logic [31:0] bytes_issued;
    logic [31:0] words_sent;
    logic [OUT_W-1:0] outstanding;

    wire ar_fire   = m_axi_arvalid && m_axi_arready;
    wire r_fire    = m_axi_rvalid  && m_axi_rready;
    wire rlast_fire= r_fire && m_axi_rlast;
    wire w_fire    = weight_valid && weight_ready;
    wire pop64     = w_fire && half_sel;

    assign m_axi_araddr  = base_addr_q + bytes_issued;
    assign m_axi_arlen   = 8'd15;
    assign m_axi_arsize  = 3'd3;      // 8 bytes / beat
    assign m_axi_arburst = 2'b01;     // INCR
    assign m_axi_arvalid = busy
                         && (bytes_issued < TOTAL_BYTES)
                         && (outstanding < MAX_OUTSTANDING);

    assign m_axi_rready = busy && (fifo_count < FIFO_DEPTH);

    assign weight_valid = busy && (fifo_count != 0);
    assign weight_data  = half_sel ? fifo_mem[rd_ptr][63:32]
                                   : fifo_mem[rd_ptr][31:0];

    assign dbg_outstanding = outstanding;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            wr_ptr       <= '0;
            rd_ptr       <= '0;
            fifo_count   <= '0;
            half_sel     <= 1'b0;
            base_addr_q  <= '0;
            bytes_issued <= '0;
            words_sent   <= '0;
            outstanding  <= '0;
            busy         <= 1'b0;
            done         <= 1'b0;
        end else begin
            done <= 1'b0;

            if (start && !busy) begin
                wr_ptr       <= '0;
                rd_ptr       <= '0;
                fifo_count   <= '0;
                half_sel     <= 1'b0;
                base_addr_q  <= base_addr;
                bytes_issued <= '0;
                words_sent   <= '0;
                outstanding  <= '0;
                busy         <= 1'b1;
            end else if (busy) begin
                if (ar_fire)
                    bytes_issued <= bytes_issued + BURST_BYTES;

                case ({ar_fire, rlast_fire})
                    2'b10: outstanding <= outstanding + 1'b1;
                    2'b01: outstanding <= outstanding - 1'b1;
                    default: outstanding <= outstanding;
                endcase

                if (r_fire) begin
                    fifo_mem[wr_ptr] <= m_axi_rdata;
                    wr_ptr <= wr_ptr + 1'b1;
                end

                if (w_fire) begin
                    words_sent <= words_sent + 1'b1;
                    if (!half_sel) begin
                        half_sel <= 1'b1;
                    end else begin
                        half_sel <= 1'b0;
                        rd_ptr <= rd_ptr + 1'b1;
                    end

                    if ((words_sent + 1'b1) == TOTAL_WORDS) begin
                        busy <= 1'b0;
                        done <= 1'b1;
                    end
                end

                case ({r_fire, pop64})
                    2'b10: fifo_count <= fifo_count + 1'b1;
                    2'b01: fifo_count <= fifo_count - 1'b1;
                    default: fifo_count <= fifo_count;
                endcase
            end
        end
    end
endmodule
