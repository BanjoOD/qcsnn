#ifndef TOP_CLASS_CBLK1_H
#define TOP_CLASS_CBLK1_H

#include <hls_stream.h>
#include "nn2_cblk1_sd.h"   // NeuralNetwork2_Cblk1_sd
#include "modeleval_sd.h"         // ModelEvaluation (fixed‑count version)
#include "../constants_sd.h"        // FIXED_LENGTH*, NUM_SAMPLES

using namespace hls4csnn1d_cblk_sd;

//---------------------------------------------------------------------
//  TopClass2 — orchestrates evaluation of NUM_SAMPLES ECG rows.
//---------------------------------------------------------------------
class TopClass2 {
public:
    TopClass2()  = default;
    ~TopClass2() = default;
    inline void instantiate() {}   // layers self‑load on first use

    void evaluate(hls::stream<ap_int8_c>& dataStream,
                  hls::stream<ap_int8_c>& labelStream)
    {
#pragma HLS INLINE
        evaluator2.evaluate(network2_cblk1_sd,
                            dataStream,
                            labelStream);
    }
private:
    NeuralNetwork2_Cblk1_sd<NUM_STEPS> network2_cblk1_sd;
    ModelEvaluation         evaluator2;
};

static void axi_to_input_data(hls::stream<axi_fixed_t>& axiStream,
                              hls::stream<ap_int8_c>&    dataStream)
{
    const int WORDS_PER_ROW = (FIXED_LENGTH1 + 7) / 8;   // 23 words
    
    for (int w = 0; w < WORDS_PER_ROW; ++w) {
#pragma HLS PIPELINE II=1  // ← Change from II=8
        axi_fixed_t word = axiStream.read();
        
        for (int j = 0; j < 8; ++j) {
#pragma HLS UNROLL
            int idx = w * 8 + j;
            if (idx < FIXED_LENGTH1) {
                ap_int8_c temp;
                temp.range(7,0) = word.data.range(j*8+7, j*8);
                dataStream.write(temp);
            }
        }
        
#ifndef __SYNTHESIS__
        if ((w == WORDS_PER_ROW - 1) && (word.last != 1)) {
            std::cerr << "[WARN] TLAST mismatch on word " << w << '\n';
        }
#endif
    }
}



static void fixed_to_axi(hls::stream<ap_int8_c>&  in,
                         hls::stream<axi_fixed_t>& out) {
    const int BYTES_PER_ROW  = 2;
    const int WORDS_PER_ROW  = 1;
#pragma HLS STREAM variable=out depth=16

    axi_fixed_t word;
    word.data = 0;
    word.keep = 0x0003;
    word.last = 1;

    for (int j = 0; j < BYTES_PER_ROW; ++j) {
#pragma HLS UNROLL
        ap_int8_c v = in.read();
        word.data.range(j*8+7, j*8) = v.range(7,0);
    }
    out.write(word);
}





//---------------------------------------------------------------------
//  Top‑level HLS function exposed to Vitis / DMA wrapper
//---------------------------------------------------------------------
extern "C" void topFunctionCblk1(hls::stream<axi_fixed_t>& dmaInStream,
                                 hls::stream<axi_fixed_t>& dmaOutStream) {
#pragma HLS INTERFACE axis      port=dmaInStream
#pragma HLS INTERFACE axis      port=dmaOutStream
// #pragma HLS INTERFACE s_axilite port=return bundle=CTRL
// In topFunctionCblk1
#pragma HLS INTERFACE ap_ctrl_hs port=return
#pragma HLS DATAFLOW   // <-- NEW: allow concurrent processes
    // Internal streams
    hls::stream<ap_int8_c> dataStream;
#pragma HLS STREAM variable=dataStream depth=32

    hls::stream<ap_int8_c> labelStream;
#pragma HLS STREAM variable=labelStream depth=32

    // 1) Unpack NUM_SAMPLES rows from AXI → dataStream
    axi_to_input_data(dmaInStream, dataStream);

    // 2) Evaluate network
    TopClass2 topClass2;
    topClass2.instantiate();          // no‑op; kept for symmetry
    topClass2.evaluate(dataStream, labelStream);

    // 3) Pack NUM_SAMPLES rows back to AXI
    fixed_to_axi(labelStream, dmaOutStream);
}

#endif // TOP_CLASS_CBLK1_H



