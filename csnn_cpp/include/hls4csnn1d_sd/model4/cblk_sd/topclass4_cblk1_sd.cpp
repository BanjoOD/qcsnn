#ifndef TOP_CLASS4_CBLK1_H
#define TOP_CLASS4_CBLK1_H

#include <hls_stream.h>
#include "nn4_cblk1_sd.h"   // NeuralNetwork2_Cblk1_sd
#include "modeleval4_sd.h"         // ModelEvaluation (fixed‑count version)
#include "../constants4_sd.h"        // FIXED_LENGTH*, NUM_SAMPLES

using namespace hls4csnn1d_cblk_sd;

//---------------------------------------------------------------------
//  TopClass2 — orchestrates evaluation of NUM_SAMPLES ECG rows.
//---------------------------------------------------------------------
class TopClass4 {
public:
    TopClass4()  = default;
    ~TopClass4() = default;
    inline void instantiate() {}   // layers self‑load on first use

    void evaluate(hls::stream<ap_int8_c>& dataStream,
                  hls::stream<ap_int8_c>& labelStream)
    {
#pragma HLS INLINE
        evaluator4.evaluate(network4_cblk1_sd,
                            dataStream,
                            labelStream);
    }
private:
    NeuralNetwork4_Cblk1_sd<NUM_STEPS> network4_cblk1_sd;
    ModelEvaluation         evaluator4;
};

// ---------------------------------------------------------------------
// 2) Topclass side: axi_fixed_t → ap_int8_c stream (axi_to_input_data)
//    Unpacks 23 AXI words back into 180 bytes.
// ---------------------------------------------------------------------
static void axi_to_input_data(hls::stream<axi_fixed_t>& axiStream,
                              hls::stream<ap_int8_c>&   dataStream) {
    const int WORDS_PER_ROW = (FIXED_LENGTH1 + 7) / 8;   // 23 words

    for (int w = 0; w < WORDS_PER_ROW; ++w) {
    #pragma HLS PIPELINE II=1
        axi_fixed_t word = axiStream.read();

        for (int j = 0; j < 8; ++j) {
        #pragma HLS UNROLL
            int idx = w * 8 + j;
            if (idx < FIXED_LENGTH1) {
                ap_int8_c temp;
                temp.range(7, 0) = word.data.range(j * 8 + 7, j * 8);
                dataStream.write(temp);
            }
        }

    #ifndef __SYNTHESIS__
        if ((w == WORDS_PER_ROW - 1) && (word.last != 1)) {
            std::cerr << "[WARN] TLAST mismatch on input word " << w << '\n';
        }
    #endif
    }
}


// ---------------------------------------------------------------------
// 3) Topclass side: ap_int8_c (4 logits) → axi_fixed_t (fixed_to_axi)
//    Packs 4×int8 logits into one 64-bit AXI word (low 4 bytes used).
// ---------------------------------------------------------------------
static void fixed_to_axi(hls::stream<ap_int8_c>&  in,
                         hls::stream<axi_fixed_t>& out) {
    const int BYTES_PER_ROW = FIXED_LENGTH9;   // 4 logits
#pragma HLS STREAM variable=out depth=16

    axi_fixed_t word;
    word.data = 0;
    word.keep = 0;
    word.last = 1;  // single word per row

    // Fill 8-byte word; only first 4 bytes come from the stream
    for (int j = 0; j < 8; ++j) {
    #pragma HLS UNROLL
        ap_uint<8> byte_val = 0;

        if (j < BYTES_PER_ROW) {
            ap_int8_c v = in.read();               // 4 reads total
            byte_val = v.range(7, 0);
            word.keep |= (ap_uint<8>)(1u << j);    // mark byte j as valid
        }

        word.data.range(j * 8 + 7, j * 8) = byte_val;
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
    TopClass4 topClass4;
    topClass4.instantiate();          // no‑op; kept for symmetry
    topClass4.evaluate(dataStream, labelStream);

    // 3) Pack NUM_SAMPLES rows back to AXI
    fixed_to_axi(labelStream, dmaOutStream);
}

#endif // TOP_CLASS_CBLK1_H



