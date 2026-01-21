#ifndef NEURAL_NETWORK2_CBLK1_SD_H
#define NEURAL_NETWORK2_CBLK1_SD_H

#include <hls_stream.h>
#include <ap_fixed.h>
#include "../constants_sd.h"
#include "includeheaders_sd.h"

/* ENABLE debug prints for *this* translation unit */
#define PRINT_FIRST_CONV_ROW
#include "conv1d_sd.h"          // ← debug code compiled in
#undef  PRINT_FIRST_CONV_ROW    // ← other files stay silent

#include "batchnorm1d_sd.h"
#include "lif1d_integer.h"
#include "maxpool1d_sd.h"
#include "linear1d_sd.h"
#include "quantidentity1d_sd.h"
#include "../utils_sd.h"          

namespace hls4csnn1d_cblk_sd {

/* ─────────────────────────────────────────────────────────────── */
template<int NUM_STEPS>
class NeuralNetwork2_Cblk1_sd {
public:
    NeuralNetwork2_Cblk1_sd()  = default;
    ~NeuralNetwork2_Cblk1_sd() = default;

/* ================================================================
 *  1.  INFERENCE FORWARD
 * ================================================================ */
    void forward(hls::stream<ap_int8_c>& input_stream,
                 hls::stream<ap_int8_c>& output_stream) {
#pragma HLS DATAFLOW
#pragma HLS INTERFACE axis      port=input_stream
#pragma HLS INTERFACE axis      port=output_stream
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

        // --- 0) Buffer ONE record so we can replay it T times ---
        ap_int8_c in_buf[CONV_IN_LENGTH1];
#pragma HLS ARRAY_PARTITION variable=in_buf cyclic factor=8 dim=1
        for (int k = 0; k < CONV_IN_LENGTH1; ++k) {
#pragma HLS PIPELINE II=1
            in_buf[k] = input_stream.read();
        }

        hls::stream<ap_int8_c> s0("conv_to_bn_cblk1");   // conv1 -> bn1
#pragma HLS STREAM variable=s0 depth=FIXED_LENGTH2

        hls::stream<ap_int8_c> s1("bn_to_lif_cblk1");    // bn1 -> lif1
#pragma HLS STREAM variable=s1 depth=FIXED_LENGTH2

        hls::stream<ap_int8_c> s2("lif_to_mp_cblk1");    // lif1 -> mp1
#pragma HLS STREAM variable=s2 depth=FIXED_LENGTH2

        hls::stream<ap_int8_c> s3("mp_to_qi_cblk2");     // mp1 -> qi(cblk2)
#pragma HLS STREAM variable=s3 depth=FIXED_LENGTH3

        hls::stream<ap_int8_c> s4("qi_to_conv_cblk2");   // qi -> conv2
#pragma HLS STREAM variable=s4 depth=FIXED_LENGTH3

        hls::stream<ap_int8_c> s5("conv_to_bn_cblk2");   // conv2 -> bn2
#pragma HLS STREAM variable=s5 depth=FIXED_LENGTH4

        hls::stream<ap_int8_c> s6("bn_to_lif_cblk2");    // bn2 -> lif2
#pragma HLS STREAM variable=s6 depth=FIXED_LENGTH4

        hls::stream<ap_int8_c> s7("lif_to_mp_cblk2");    // lif2 -> mp2
#pragma HLS STREAM variable=s7 depth=FIXED_LENGTH4

        hls::stream<ap_int8_c> s8("mp_to_qi_cblk3");     // mp2 -> qi(head)
#pragma HLS STREAM variable=s8 depth=FIXED_LENGTH5

        hls::stream<ap_int8_c> s9("qi_to_fc_lblk1");     // qi -> linear
#pragma HLS STREAM variable=s9 depth=FIXED_LENGTH5

        hls::stream<ap_int8_c> s10("fc_to_lif_lblk1");   // linear -> lif(head)
#pragma HLS STREAM variable=s10 depth=FIXED_LENGTH6

        // internal stream for THIS step’s head-LIF output
        hls::stream<ap_int8_c> s_out_step("lif_out_step");
#pragma HLS STREAM variable=s_out_step depth=4

        // reset LIF states ONCE per record (match utils.reset(net))
        qcsnet2_cblk1_lif1d_int.reset();
        qcsnet2_cblk2_lif1d_int.reset();
        qcsnet2_lblk1_lif1d_int.reset();

        // widen accumulators for spike counts (two output neurons)
        ap_int<16> sum0 = 0, sum1 = 0;

REPEAT_STEPS:
        for (int t = 0; t < NUM_STEPS; ++t) {
            // (a) re-stream the SAME buffered input into a small local FIFO
            hls::stream<ap_int8_c> s_in("in_replay");
#pragma HLS STREAM variable=s_in depth=CONV_IN_LENGTH1
            for (int k = 0; k < CONV_IN_LENGTH1; ++k) {
#pragma HLS PIPELINE II=1
                s_in.write(in_buf[k]);
            }

            // cblk1: conv → bn → lif → mp
            qcsnet2_cblk1_qconv1d_int.forward(
                s_in, s0,
                qcsnet2_cblk1_qconv1d_weights,
                qcsnet2_cblk1_qconv1d_scale_multiplier,
                qcsnet2_cblk1_qconv1d_right_shift,
                qcsnet2_cblk1_qconv1d_bias,
                qcsnet2_cblk1_qconv1d_input_zero_point,
                qcsnet2_cblk1_qconv1d_weight_sum);

            qcsnet2_cblk1_bn1d_int.forward(
                s0, s1,
                qcsnet2_cblk1_batch_norm_weight,
                qcsnet2_cblk1_batch_norm_bias,
                qcsnet2_cblk1_batch_norm_scale_multiplier,
                qcsnet2_cblk1_batch_norm_right_shift);

            qcsnet2_cblk1_lif1d_int.forward(
                s1, s2,
                qcsnet2_cblk1_leaky_beta_int,
                qcsnet2_cblk1_leaky_theta_int,
                qcsnet2_cblk1_leaky_scale_int);

            qcsnet2_cblk1_maxpool1d.forward(s2, s3);

            // quant identity before next conv block (maps {0,1}->{0,q_one})
            qcsnet2_cblk2_qi_int.forward(
                s3, s4,
                qcsnet2_cblk2_input_act_scale_int);

            // cblk2: conv → bn → lif → mp
            qcsnet2_cblk2_qconv1d_int.forward(
                s4, s5,
                qcsnet2_cblk2_qconv1d_weights,
                qcsnet2_cblk2_qconv1d_scale_multiplier,
                qcsnet2_cblk2_qconv1d_right_shift,
                qcsnet2_cblk2_qconv1d_bias,
                qcsnet2_cblk2_qconv1d_input_zero_point,
                qcsnet2_cblk2_qconv1d_weight_sum);

            qcsnet2_cblk2_bn1d_int.forward(
                s5, s6,
                qcsnet2_cblk2_batch_norm_weight,
                qcsnet2_cblk2_batch_norm_bias,
                qcsnet2_cblk2_batch_norm_scale_multiplier,
                qcsnet2_cblk2_batch_norm_right_shift);

            qcsnet2_cblk2_lif1d_int.forward(
                s6, s7,
                qcsnet2_cblk2_leaky_beta_int,
                qcsnet2_cblk2_leaky_theta_int,
                qcsnet2_cblk2_leaky_scale_int);

            qcsnet2_cblk2_maxpool1d.forward(s7, s8);

            // quant identity before linear head
            qcsnet2_lblk1_qi_int.forward(
                s8, s9,
                qcsnet2_lblk1_input_act_scale_int);

            // dense head: linear → lif
            qcsnet2_lblk1_qlinear1d_int.forward(
                s9, s10,
                qcsnet2_lblk1_qlinear_weights,
                qcsnet2_lblk1_qlinear_scale_multiplier,
                qcsnet2_lblk1_qlinear_right_shift,
                qcsnet2_lblk1_qlinear_bias,
                qcsnet2_lblk1_qlinear_input_zero_point,
                qcsnet2_lblk1_qlinear_weight_sum);

            qcsnet2_lblk1_lif1d_int.forward(
                s10, s_out_step,
                qcsnet2_lblk1_leaky_beta_int,
                qcsnet2_lblk1_leaky_theta_int,
                qcsnet2_lblk1_leaky_scale_int);

            // (c) consume this step’s two outputs and accumulate counts
            ap_int8_c y0 = s_out_step.read();  // {0,1}
            ap_int8_c y1 = s_out_step.read();  // {0,1}
            sum0 = sum0 + (ap_int<16>)y0;
            sum1 = sum1 + (ap_int<16>)y1;
        }

        // (d) write the averaged pair once (Q12 fixed-point average)
        const int F = 12;
        ap_int<16> invT_q = ((ap_int<16>)(1 << F) + (ap_int<16>)(NUM_STEPS >> 1))
                            / (ap_int<16>)NUM_STEPS;

        auto avg_q12_to_i8 = [&](ap_int<16> s)->ap_int8_c {
            ap_int<32> p = (ap_int<32>)s * (ap_int<32>)invT_q; // Q12 product
            p += (ap_int<32>)(1 << (F - 1));                   // round to nearest
            ap_int<16> a = (ap_int<16>)(p >> F);               // back to int
            if (a > 127)  return ap_int8_c(127);
            if (a < -128) return ap_int8_c(-128);
            return (ap_int8_c)a;
        };

        output_stream.write( avg_q12_to_i8(sum0) );
        output_stream.write( avg_q12_to_i8(sum1) );
    }

/* ================================================================
 *  3.  LAYER OBJECTS  (unchanged names)
 * ================================================================ */
private:
    /* 1st block */
    Conv1D_SD<1, 16, 3, 1, CONV_IN_LENGTH1>               qcsnet2_cblk1_qconv1d_int;
    BatchNorm1D_SD<OUT_CH1, FEATURE_LENGTH1>              qcsnet2_cblk1_bn1d_int;
    LIF1D_SD_Integer<OUT_CH1, FEATURE_LENGTH1>            qcsnet2_cblk1_lif1d_int;
    MaxPool1D_SD<2, 2, OUT_CH1, FEATURE_LENGTH1>          qcsnet2_cblk1_maxpool1d;

    /* 2nd block */
    QuantIdentityPerTensor_Int8<OUT_CH1, CONV_IN_LENGTH2> qcsnet2_cblk2_qi_int;
    Conv1D_SD<16, 24, 3, 1, CONV_IN_LENGTH2>              qcsnet2_cblk2_qconv1d_int;
    BatchNorm1D_SD<OUT_CH2, FEATURE_LENGTH2>              qcsnet2_cblk2_bn1d_int;
    LIF1D_SD_Integer<OUT_CH2, FEATURE_LENGTH2>            qcsnet2_cblk2_lif1d_int;
    MaxPool1D_SD<2, 2, OUT_CH2, FEATURE_LENGTH2>          qcsnet2_cblk2_maxpool1d;

    QuantIdentityPerTensor_Int8<1, LINEAR_IN_SIZE>        qcsnet2_lblk1_qi_int;

    /* dense head */
    Linear1D_SD<LINEAR_IN_SIZE, 2>                        qcsnet2_lblk1_qlinear1d_int;
    LIF1D_SD_Integer<2, 1>                                qcsnet2_lblk1_lif1d_int;
};

} // namespace hls4csnn1d_cblk_sd
#endif /* NEURAL_NETWORK2_CBLK1_SD_H */





// // #ifndef NEURAL_NETWORK2_CBLK1_SD_H
// // #define NEURAL_NETWORK2_CBLK1_SD_H

// // #include <hls_stream.h>
// // #include <ap_fixed.h>
// // #include "../constants_sd.h"
// // #include "includeheaders_sd.h"


// // /* ENABLE debug prints for *this* translation unit */
// // #define PRINT_FIRST_CONV_ROW
// // #include "conv1d_sd.h"          // ← debug code compiled in
// // #undef  PRINT_FIRST_CONV_ROW    // ← other files stay silent

// // #include "batchnorm1d_sd.h"
// // #include "lif1d_integer.h"
// // #include "maxpool1d_sd.h"
// // #include "linear1d_sd.h"
// // #include "quantidentity1d_sd.h"
// // #include "../utils_sd.h"          // vec2stream / stream2vec

// // namespace hls4csnn1d_cblk_sd {

// // /* ─────────────────────────────────────────────────────────────── */
// // template<int NUM_STEPS>
// // class NeuralNetwork2_Cblk1_sd {
// // public:
// //     NeuralNetwork2_Cblk1_sd()  = default;
// //     ~NeuralNetwork2_Cblk1_sd() = default;

// // /* ================================================================
// //  *  1.  INFERENCE FORWARD  (your original code – untouched)
// //  * ================================================================ */
// //     void forward(hls::stream<ap_int8_c>& input_stream,
// //                  hls::stream<ap_int8_c>& output_stream) {
// // #pragma HLS DATAFLOW
// // #pragma HLS INTERFACE axis      port=input_stream
// // #pragma HLS INTERFACE axis      port=output_stream
// // #pragma HLS INTERFACE s_axilite port=return bundle=CTRL

// //  		// --- 0) Buffer ONE record so we can replay it T times ---
// //     	ap_int8_c in_buf[CONV_IN_LENGTH1];
// // #pragma HLS ARRAY_PARTITION variable=in_buf cyclic factor=8 dim=1
// //     	for (int k = 0; k < CONV_IN_LENGTH1; ++k) {
// // #pragma HLS PIPELINE II=1
// //         in_buf[k] = input_stream.read();
// //     	}
 
// //         hls::stream<ap_int8_c> s0("conv_to_bn_cblk1");          // between conv1 and bn
// // #pragma HLS STREAM variable=s0 depth=FIXED_LENGTH2  // must hold up to CONV1_OUT_LEN×CONV1_OUT_CHANNEL

// //         hls::stream<ap_int8_c> s1("bn_to_lif_cblk1");          // between conv1 and bn
// // #pragma HLS STREAM variable=s1 depth=FIXED_LENGTH2

// //         hls::stream<ap_int8_c> s2("lif_to_mp_cblk1");          // between lif and maxpool
// // #pragma HLS STREAM variable=s2 depth=FIXED_LENGTH2 

// //         hls::stream<ap_int8_c> s3("mp_to_qi_cblk2");          // between lif and maxpool
// // #pragma HLS STREAM variable=s3 depth=FIXED_LENGTH3

// //         hls::stream<ap_int8_c> s4("qi_to_conv_cblk2");
// // #pragma HLS STREAM variable=s4 depth=FIXED_LENGTH3

// //         hls::stream<ap_int8_c> s5("conv_to_bn_cblk2");
// // #pragma HLS STREAM variable=s5 depth=FIXED_LENGTH4

// //         hls::stream<ap_int8_c> s6("bn_to_lif_cblk2");
// // #pragma HLS STREAM variable=s6 depth=FIXED_LENGTH4

// //         hls::stream<ap_int8_c> s7("lif_to_mp_cblk2");
// // #pragma HLS STREAM variable=s7 depth=FIXED_LENGTH4

// //         hls::stream<ap_int8_c> s8("mp_to_qi_cblk3");          // between lif and maxpool
// // #pragma HLS STREAM variable=s8 depth=FIXED_LENGTH5

// //         hls::stream<ap_int8_c> s9("qi_to_fc_lblk1");
// // #pragma HLS STREAM variable=s9 depth=FIXED_LENGTH5

// //         hls::stream<ap_int8_c> s10("fc_to_lif_lblk1");
// // #pragma HLS STREAM variable=s10 depth=FIXED_LENGTH6


// // 		// NEW: internal stream for *this step’s* head-LIF output (don’t read output port)
// //    		hls::stream<ap_int8_c> s_out_step("lif_out_step");
// // #pragma HLS STREAM variable=s_out_step depth=4


// // 		// optional: reset LIF states ONCE per record, not per step
// // 		qcsnet2_cblk1_lif1d_int.reset();
// // 		qcsnet2_cblk2_lif1d_int.reset();
// // 		qcsnet2_lblk1_lif1d_int.reset();

// // 		// accumulators for the two output neurons
// // 		ap_int8_c sum0 = ap_int8_c(0), sum1 = ap_int8_c(0);

// // REPEAT_STEPS:
// //     	for (int t = 0; t < NUM_STEPS; ++t) {
// // 			// (a) re-stream the SAME buffered input into a small local FIFO
// //         	hls::stream<ap_int8_c> s_in("in_replay");
// // #pragma HLS STREAM variable=s_in depth=CONV_IN_LENGTH1
// //         	for (int k = 0; k < CONV_IN_LENGTH1; ++k) {
// // #pragma HLS PIPELINE II=1
// //             	s_in.write(in_buf[k]);
// //         	}

		
// //         // Single‑layer pipeline for now.  Layer itself loads its weights
// //         // on the PS side the first time it executes.
// // // #ifndef __SYNTHESIS__
// // //         std::cout << "Before CONV: conv_to_bn_cblk1 size = " << s0.size() << std::endl;
// // // #endif
// //         	qcsnet2_cblk1_qconv1d_int.forward(s_in, s0, 
// //                 qcsnet2_cblk1_qconv1d_weights,
// //                 qcsnet2_cblk1_qconv1d_scale_multiplier,
// //                 qcsnet2_cblk1_qconv1d_right_shift,
// //                 qcsnet2_cblk1_qconv1d_bias,
// //                 qcsnet2_cblk1_qconv1d_input_zero_point,
// //                 qcsnet2_cblk1_qconv1d_weight_sum);
// // // #ifndef __SYNTHESIS__
// // //         std::cout << "After CONV: conv_to_bn_cblk1 size = " << s0.size() << std::endl;
// // // #endif
// // // #ifndef __SYNTHESIS__
// // //         std::cout << "Before BN1: conv_to_bn_cblk1 size = " << s0.size() << std::endl;
// // // #endif
// // 			qcsnet2_cblk1_bn1d_int.forward(s0, s1, 
// // 				qcsnet2_cblk1_batch_norm_weight, 
// // 				qcsnet2_cblk1_batch_norm_bias,
// // 				qcsnet2_cblk1_batch_norm_scale_multiplier,
// // 				qcsnet2_cblk1_batch_norm_right_shift);
// // // #ifndef __SYNTHESIS__
// // //         std::cout << "After BN1: conv_to_bn_cblk1 size = " << s0.size() << std::endl;
// // // #endif

// // // #ifdef __SYNTHESIS__
// // //         qcsnet2_cblk1_lif1d.forward(s0, s1,qcsnet2_cblk1_leaky_beta, qcsnet2_cblk1_leaky_threshold);
// // // #else
// // 			qcsnet2_cblk1_lif1d_int.forward(s1, s2, 
// //                 qcsnet2_cblk1_leaky_beta_int,
// //                 qcsnet2_cblk1_leaky_theta_int,
// //                 qcsnet2_cblk1_leaky_scale_int);
// // // #endif

// //         	// qcsnet2_cblk1_lif1d.forward(s0, s1, qcsnet2_cblk1_leaky_beta, qcsnet2_cblk1_leaky_threshold);
// // 			qcsnet2_cblk1_maxpool1d.forward(s2, s3);

// //             qcsnet2_cblk2_qi_int.forward(s3, s4);

// // 			qcsnet2_cblk2_qconv1d_int.forward(s4, s5, 
// //                 qcsnet2_cblk2_qconv1d_weights,
// //                 qcsnet2_cblk2_qconv1d_scale_multiplier,
// //                 qcsnet2_cblk2_qconv1d_right_shift,
// //                 qcsnet2_cblk2_qconv1d_bias,
// //                 qcsnet2_cblk2_qconv1d_input_zero_point,
// //                 qcsnet2_cblk2_qconv1d_weight_sum);
// // // #ifndef __SYNTHESIS__
// // //         std::cout << "Before BN2: conv_to_bn_cblk2 size = " << s4.size() << std::endl;
// // // #endif
// // 			qcsnet2_cblk2_bn1d_int.forward(s5, s6, 
// //                 qcsnet2_cblk2_batch_norm_weight, 
// // 				qcsnet2_cblk2_batch_norm_bias,
// // 				qcsnet2_cblk2_batch_norm_scale_multiplier,
// // 				qcsnet2_cblk2_batch_norm_right_shift);
// // // #ifndef __SYNTHESIS__
// // //         std::cout << "After BN2: conv_to_bn_cblk2 size = " << s4.size() << std::endl;
// // // #endif
// // // --- LIF in block 2 ---
// // // #ifdef __SYNTHESIS__
// // //         qcsnet2_cblk2_lif1d.forward(s3, s4, qcsnet2_cblk2_leaky_beta, qcsnet2_cblk2_leaky_threshold);
// // // #else
// // 			qcsnet2_cblk2_lif1d_int.forward(s6, s7, 
// //                 qcsnet2_cblk2_leaky_beta_int,
// //                 qcsnet2_cblk2_leaky_theta_int,
// //                 qcsnet2_cblk2_leaky_scale_int);
// // // #endif
// //         // qcsnet2_cblk2_lif1d.forward(s3, s4, qcsnet2_cblk2_leaky_beta, qcsnet2_cblk2_leaky_threshold);
// // 			qcsnet2_cblk2_maxpool1d.forward(s7, s8);


// //             qcsnet2_lblk1_qi_int.forward(s8, s9);

// // 			qcsnet2_lblk1_qlinear1d_int.forward(s9, s10, 
// //                 qcsnet2_lblk1_qlinear_weights,
// //                 qcsnet2_lblk1_qlinear_scale_multiplier,
// //                 qcsnet2_lblk1_qlinear_right_shift,
// //                 qcsnet2_lblk1_qlinear_bias,
// //                 qcsnet2_lblk1_qlinear_input_zero_point,
// //                 qcsnet2_lblk1_qlinear_weight_sum);
// //         // --- LIF in linear head ---
// // // #ifdef __SYNTHESIS__
// // //         qcsnet2_lblk1_lif1d.forward(s6, output_stream, qcsnet2_lblk1_leaky_beta, qcsnet2_lblk1_leaky_threshold);
// // // #else
            
// // 			qcsnet2_lblk1_lif1d_int.forward(s10, s_out_step, 
// //                 qcsnet2_lblk1_leaky_beta_int,
// //                 qcsnet2_lblk1_leaky_theta_int,
// //                 qcsnet2_lblk1_leaky_scale_int);
// // // #endif

// //         // qcsnet2_lblk1_lif1d.forward(s6, output_stream, qcsnet2_lblk1_leaky_beta, qcsnet2_lblk1_leaky_threshold);

// // 		// (c) consume this step’s two outputs and accumulate
// // 			ap_int8_c y0 = s_out_step.read();
// // 			ap_int8_c y1 = s_out_step.read();
// // 			sum0 += y0;
// // 			sum1 += y1;

// // 		}
// // 		// (d) write the *averaged* pair once to the real output port
// // 		ap_int8_c invT = ap_int8_c(1) / ap_int8_c(NUM_STEPS);
// // 		output_stream.write(sum0 * invT);
// // 		output_stream.write(sum1 * invT);

// //     }

// // /* ================================================================
// //  *  3.  LAYER OBJECTS  (unchanged)
// //  * ================================================================ */
// // private:
// //     /* 1st block */
// //     Conv1D_SD<1, 16, 3, 1, CONV_IN_LENGTH1>       qcsnet2_cblk1_qconv1d_int;
// //     BatchNorm1D_SD<OUT_CH1, FEATURE_LENGTH1>         qcsnet2_cblk1_bn1d_int;
// //     LIF1D_SD_Integer<OUT_CH1, FEATURE_LENGTH1>           qcsnet2_cblk1_lif1d_int;
// //     MaxPool1D_SD<2, 2, OUT_CH1, FEATURE_LENGTH1>     qcsnet2_cblk1_maxpool1d;

// //     /* 2nd block */
// //     QuantIdentityPerTensor_Int8<OUT_CH2, CONV_IN_LENGTH2>     qcsnet2_cblk2_qi_int;
// //     Conv1D_SD<16, 24, 3, 1, CONV_IN_LENGTH2>      qcsnet2_cblk2_qconv1d_int;
// //     BatchNorm1D_SD<OUT_CH2, FEATURE_LENGTH2>         qcsnet2_cblk2_bn1d_int;
// //     LIF1D_SD_Integer<OUT_CH2, FEATURE_LENGTH2>           qcsnet2_cblk2_lif1d_int;
// //     MaxPool1D_SD<2, 2, OUT_CH2, FEATURE_LENGTH2>     qcsnet2_cblk2_maxpool1d;

// //     QuantIdentityPerTensor_Int8<LINEAR_IN_SIZE, 2>     qcsnet2_lblk1_qi_int;

// //     /* dense head */
// //     Linear1D_SD <LINEAR_IN_SIZE, 2>                  qcsnet2_lblk1_qlinear1d_int;
// //     LIF1D_SD_Integer   <2, 1>                               qcsnet2_lblk1_lif1d_int;
// // };

// // } // namespace hls4csnn1d_cblk_sd
// // #endif /* NEURAL_NETWORK2_CBLK1_SD_H */
