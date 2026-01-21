#ifndef MODEL_EVAL4_REF_H
#define MODEL_EVAL4_REF_H

#include "../constants4_sd.h"
#include "hls_stream.h"
#include "ap_fixed.h"
#include "nn4_cblk1_sd.h"
#include "../filereader4.h"

// Refactored ModelEvaluation: fixed-count loops to avoid hanging in co-sim/synthesis
class ModelEvaluation {
public:
    /**
     * Evaluate a fixed number of samples through the model.
     * @param model4      Neural network instance
     * @param datastream  Stream of input samples (each input180_data_type)
     * @param labelstream Stream to write output samples (each input240_data_type)
     */
    //  stream‑level evaluator  (all FIFOs carry ap_int8_c)
void evaluate(hls4csnn1d_cblk_sd::NeuralNetwork4_Cblk1_sd<NUM_STEPS>& model4,
              hls::stream<ap_int8_c>& datastream,          // scalar IN
              hls::stream<ap_int8_c>& labelstream)         // scalar OUT
{
#pragma HLS INLINE


    // sample_loop:
    // for (int row = 0; row < num_samples; ++row) {
        //------------------------------------------------------------------
        // 1) Collect one input row (180 scalars) → input_stream
        //------------------------------------------------------------------
    hls::stream<ap_int8_c> input_stream;
#pragma HLS STREAM variable=input_stream depth=FIXED_LENGTH1  // 180
        
    // local capture buffer for saving (1×180)
    array180_t row_buf;

    in_write:
    for (int i = 0; i < FIXED_LENGTH1; ++i) {
#pragma HLS PIPELINE II=1
        ap_int8_c v = datastream.read();
        input_stream.write(v);
        row_buf[i] = v;   // capture exactly what goes into input_stream
    }

        // ---- (optional for synthesis safety) dump to CSV in CSIM only ----
        // You can remove the ifdef if you want CSIM *and* SW emulation dumps.
// #ifndef __SYNTHESIS__
        // {
        //     // choose a directory & prefix for these evaluator dumps
        //     const std::string outdir = "quantized_inputs_eval";
        //     const std::string prefix = "eval_in";

        //     FileReader fileReader;

        //     // idx = 1-based running count within this call
        //     fileReader.saveRowToCSV(row_buf, outdir, prefix, static_cast<size_t>(row + 1));
        // }
// #endif

    //------------------------------------------------------------------
    // 2) Forward through the model  → output_stream (2 scalars)
    //------------------------------------------------------------------
    hls::stream<ap_int8_c> output_stream;
#pragma HLS STREAM variable=output_stream depth=FIXED_LENGTH9 // 4

    model4.forward(input_stream, output_stream);

    //------------------------------------------------------------------
    // 3) Push all 2 scalars into labelstream
    //------------------------------------------------------------------
    out_write:
    for (int i = 0; i < FIXED_LENGTH9; ++i) {
#pragma HLS PIPELINE II=1
        labelstream.write(output_stream.read());
    }
    // }
}

// inline void checksum_row(const array180_t& row, double& sum, double& sumsq, double& maxabs) {
//     sum = 0.0; sumsq = 0.0; maxabs = 0.0;
//     for (int k = 0; k < FIXED_LENGTH1; ++k) {
//         float v = (float)row[k];
//         sum   += v;
//         sumsq += v * v;
//         float a = std::fabs(v);
//         if (a > maxabs) maxabs = a;
//     }
// }

};

#endif // MODEL_EVAL_REF_H



