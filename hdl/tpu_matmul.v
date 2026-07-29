// TPU Systolic Array Matrix Multiplication Unit
module tpu_matmul #(
    parameter DATA_WIDTH = 16,
    parameter ARRAY_SIZE = 128
) (
    input wire clk,
    input wire reset_n,
    input wire enable,
    input wire [DATA_WIDTH-1:0] act_in,
    input wire [DATA_WIDTH-1:0] weight_in,
    output reg [DATA_WIDTH*2-1:0] accum_out
);

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            accum_out <= 0;
        end else if (enable) begin
            accum_out <= accum_out + (act_in * weight_in);
        end
    end

endmodule
