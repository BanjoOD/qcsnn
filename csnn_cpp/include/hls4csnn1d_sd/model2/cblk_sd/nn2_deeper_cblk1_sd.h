// #ifndef NEURAL_NETWORK2_DEEPER_CBLK1_SD_H
// #define NEURAL_NETWORK2_DEEPER_CBLK1_SD_H

// #include <hls_stream.h>
// #include <ap_fixed.h>
// #include "../constants_sd.h"
// #include "includeheaders_sd.h"


// /* ENABLE debug prints for *this* translation unit */
// #define PRINT_FIRST_CONV_ROW
// #include "conv1d_sd.h"          // ← debug code compiled in
// #undef  PRINT_FIRST_CONV_ROW    // ← other files stay silent

// // #include "batchnorm1d_sd.h"
// #include "lif1d_sd.h"
// #include "maxpool1d_sd.h"
// #include "linear1d_sd.h"
// #include "../utils_sd.h"          // vec2stream / stream2vec

// namespace hls4csnn1d_cblk_sd {

// /* ─────────────────────────────────────────────────────────────── */
// template<int NUM_STEPS>
// class NeuralNetwork2_Cblk1_sd {
// public:
//     NeuralNetwork2_Cblk1_sd()  = default;
//     ~NeuralNetwork2_Cblk1_sd() = default;

// /* ================================================================
//  *  1.  INFERENCE FORWARD  (your original code – untouched)
//  * ================================================================ */
//     void forward(hls::stream<ap_fixed_c>& input_stream,
//                  hls::stream<ap_fixed_c>& output_stream) {
// #pragma HLS DATAFLOW
// #pragma HLS INTERFACE axis      port=input_stream
// #pragma HLS INTERFACE axis      port=output_stream
// #pragma HLS INTERFACE s_axilite port=return bundle=CTRL

//  		// --- 0) Buffer ONE record so we can replay it T times ---
//     	ap_fixed_c in_buf[CONV_IN_LENGTH1];
// #pragma HLS ARRAY_PARTITION variable=in_buf cyclic factor=8 dim=1
//     	for (int k = 0; k < CONV_IN_LENGTH1; ++k) {
// #pragma HLS PIPELINE II=1
//         in_buf[k] = input_stream.read();
//     	}

//         hls::stream<ap_fixed_c> s0("conv_to_lif_cblk1");          // between conv1 and bn 2136
// #pragma HLS STREAM variable=s0 depth=FIXED_LENGTH2  // must hold up to CONV1_OUT_LEN×CONV1_OUT_CHANNEL

//         hls::stream<ap_fixed_c> s1("lif_to_mp_cblk1");          // between lif and maxpool 2136
// #pragma HLS STREAM variable=s1 depth=FIXED_LENGTH2 

//         hls::stream<ap_fixed_c> s2("mp_to_conv_cblk2"); //1068
// #pragma HLS STREAM variable=s2 depth=FIXED_LENGTH3

//         hls::stream<ap_fixed_c> s3("conv_to_lif_cblk2");
// #pragma HLS STREAM variable=s3 depth=FIXED_LENGTH4

//         hls::stream<ap_fixed_c> s4("lif_to_conv_cblk3");
// #pragma HLS STREAM variable=s4 depth=FIXED_LENGTH4

//         hls::stream<ap_fixed_c> s5("conv_to_lif_cblk3");
// #pragma HLS STREAM variable=s5 depth=FIXED_LENGTH5

//         hls::stream<ap_fixed_c> s6("lif_to_mp_cblk3");
// #pragma HLS STREAM variable=s6 depth=FIXED_LENGTH5

//         hls::stream<ap_fixed_c> s7("mp_to_conv_cblk4");
// #pragma HLS STREAM variable=s7 depth=FIXED_LENGTH6

//         hls::stream<ap_fixed_c> s8("conv_to_lif_cblk4");
// #pragma HLS STREAM variable=s8 depth=FIXED_LENGTH7

//         hls::stream<ap_fixed_c> s9("lif_to_fc_lblk1");
// #pragma HLS STREAM variable=s9 depth=FIXED_LENGTH7

//         hls::stream<ap_fixed_c> s10("fc_to_lif_lblk1");
// #pragma HLS STREAM variable=s10 depth=FIXED_LENGTH8


// 		// NEW: internal stream for *this step’s* head-LIF output (don’t read output port)
//    		hls::stream<ap_fixed_c> s_out_step("lif_out_step");
// #pragma HLS STREAM variable=s_out_step depth=FIXED_LENGTH8


// 		// optional: reset LIF states ONCE per record, not per step
// 		qcsnet2_cblk1_lif1d.reset();
// 		qcsnet2_cblk2_lif1d.reset();
// 		qcsnet2_lblk1_lif1d.reset();

// 		// accumulators for the two output neurons
// 		ap_fixed_c sum0 = ap_fixed_c(0), sum1 = ap_fixed_c(0);

// REPEAT_STEPS:
//     	for (int t = 0; t < NUM_STEPS; ++t) {
// 			// (a) re-stream the SAME buffered input into a small local FIFO
//         	hls::stream<ap_fixed_c> s_in("in_replay");
// #pragma HLS STREAM variable=s_in depth=CONV_IN_LENGTH1
//         	for (int k = 0; k < CONV_IN_LENGTH1; ++k) {
// #pragma HLS PIPELINE II=1
//             	s_in.write(in_buf[k]);
//         	}
// // #ifndef __SYNTHESIS__
// //         std::cout << "Before CBLK1 CONV : s_in size = " << s_in.size() << std::endl;
// // #endif

//             qcsnet2_cblk1_qconv1d_folded.forward(s_in, s0, qcsnet2_cblk1_qconv_weights, qcsnet2_cblk1_qconv_bias);
// // #ifndef __SYNTHESIS__
// //         std::cout << "After CBLK1 CONV : s_in size = " << s_in.size() << std::endl;
// // #endif

// // #ifndef __SYNTHESIS__
// //         std::cout << "Before CBLK1 LIF : lif_to_mp_cblk1 size = " << s1.size() << std::endl;
// // #endif
// 			qcsnet2_cblk1_lif1d.forward(s0, s1, qcsnet2_cblk1_leaky_beta, qcsnet2_cblk1_leaky_theta);
// // #ifndef __SYNTHESIS__
// //         std::cout << "After CBLK1 LIF : lif_to_mp_cblk1 size = " << s1.size() << std::endl;
// // #endif

// // #ifndef __SYNTHESIS__
// //         std::cout << "Before CBLK1 MP : mp_to_conv_cblk2 size = " << s2.size() << std::endl;
// // #endif
// 			qcsnet2_cblk1_maxpool1d.forward(s1, s2);
// // #ifndef __SYNTHESIS__
// //         std::cout << "After CBLK1 MP : mp_to_conv_cblk2 size = " << s2.size() << std::endl;
// // #endif

// // #ifndef __SYNTHESIS__
// //         std::cout << "Before CBLK2 CONV: conv_to_lif_cblk2 size = " << s3.size() << std::endl;
// // #endif		
//             qcsnet2_cblk2_qconv1d_folded.forward(s2, s3, qcsnet2_cblk2_qconv_weights, qcsnet2_cblk2_qconv_bias);
// // #ifndef __SYNTHESIS__
// //         std::cout << "After CBLK2 CONV : conv_to_lif_cblk2 size = " << s3.size() << std::endl;
// // #endif

// // #ifndef __SYNTHESIS__
// //         std::cout << "Before CBLK2 LIF: lif_to_conv_cblk3 size = " << s4.size() << std::endl;
// // #endif	
// 			qcsnet2_cblk2_lif1d.forward(s3, s4, qcsnet2_cblk2_leaky_beta, qcsnet2_cblk2_leaky_theta);
// // #ifndef __SYNTHESIS__
// //         std::cout << "After CBLK2 LIF : lif_to_conv_cblk3 size = " << s4.size() << std::endl;
// // #endif

// // #ifndef __SYNTHESIS__
// //         std::cout << "Before CBLK3 CONV: conv_to_lif_cblk3 size = " << s5.size() << std::endl;
// // #endif
//             qcsnet2_cblk3_qconv1d_folded.forward(s4, s5, qcsnet2_cblk3_qconv_weights, qcsnet2_cblk3_qconv_bias);
// // #ifndef __SYNTHESIS__
// //         std::cout << "After CBLK3 CONV : conv_to_lif_cblk3 size = " << s5.size() << std::endl;
// // #endif
// // #ifndef __SYNTHESIS__
// //         std::cout << "Before CBLK3 LIF: lif_to_mp_cblk3 size = " << s6.size() << std::endl;
// // #endif	
// 			qcsnet2_cblk3_lif1d.forward(s5, s6, qcsnet2_cblk3_leaky_beta, qcsnet2_cblk3_leaky_theta);
// // #ifndef __SYNTHESIS__
// //         std::cout << "After CBLK3 LIF : lif_to_mp_cblk3 size = " << s6.size() << std::endl;
// // #endif
// // #ifndef __SYNTHESIS__
// //         std::cout << "Before CBLK3 MP : mp_to_conv_cblk4 size = " << s7.size() << std::endl;
// // #endif
// 			qcsnet2_cblk3_maxpool1d.forward(s6, s7);
// // #ifndef __SYNTHESIS__
// //         std::cout << "After CBLK3 MP : mp_to_conv_cblk4 size = " << s7.size() << std::endl;
// // #endif

// // #ifndef __SYNTHESIS__
// //         std::cout << "Before CBLK4 CONV: conv_to_lif_cblk4 size = " << s8.size() << std::endl;
// // #endif
// 			qcsnet2_cblk4_qconv1d_folded.forward(s7, s8, qcsnet2_cblk4_qconv_weights, qcsnet2_cblk4_qconv_bias);
// // #ifndef __SYNTHESIS__
// //         std::cout << "After CBLK4 CONV: conv_to_lif_cblk4 size = " << s8.size() << std::endl;
// // #endif
// // #ifndef __SYNTHESIS__
// //         std::cout << "Before CBLK4 CONV: lif_to_linear_lblk1 size = " << s9.size() << std::endl;
// // #endif
// 			qcsnet2_cblk4_lif1d.forward(s8, s9, qcsnet2_cblk4_leaky_beta, qcsnet2_cblk4_leaky_theta);
// // #ifndef __SYNTHESIS__
// //         std::cout << "After CBLK4 CONV: lif_to_linear_lblk1 size = " << s9.size() << std::endl;
// // #endif
// // #ifndef __SYNTHESIS__
// //         std::cout << "Before LBLK1 LINEAR: linear_to_lif_lblk1 size = " << s10.size() << std::endl;
// // #endif
// 			qcsnet2_lblk1_qlinear1d.forward(s9, s10, qcsnet2_lblk1_qlinear_weights);
// // #ifndef __SYNTHESIS__
// //         std::cout << "After LBLK1 LINEAR: linear_to_lif_lblk1 size = " << s10.size() << std::endl;
// // #endif
// // #ifndef __SYNTHESIS__
// //         std::cout << "Before LBLK1 LIF: out size = " << s_out_step.size() << std::endl;
// // #endif
// 			qcsnet2_lblk1_lif1d.forward(s10, s_out_step, qcsnet2_lblk1_leaky_beta, qcsnet2_lblk1_leaky_theta);
// // #ifndef __SYNTHESIS__
// //         std::cout << "After LBLK1 LIF: out size = " << s_out_step.size() << std::endl;
// // #endif

// // #ifndef __SYNTHESIS__
// //         std::cout << "Before y0: out size = " << s_out_step.size() << std::endl;
// // #endif
// 		// (c) consume this step’s two outputs and accumulate
// 			ap_fixed_c y0 = s_out_step.read();
// // #ifndef __SYNTHESIS__
// //         std::cout << "After y0: out size = " << s_out_step.size() << std::endl;
// // #endif
// // #ifndef __SYNTHESIS__
// //         std::cout << "Before y1: out size = " << s_out_step.size() << std::endl;
// // #endif
// 			ap_fixed_c y1 = s_out_step.read();
// // #ifndef __SYNTHESIS__
// //         std::cout << "After y1: out size = " << s_out_step.size() << std::endl;
// // #endif
// 			sum0 += y0;
// 			sum1 += y1;

// 		}
// // #ifndef __SYNTHESIS__
// //         std::cout << "Before : out_stream size = " << output_stream.size() << std::endl;
// // #endif
// 		// (d) write the *averaged* pair once to the real output port
// 		ap_fixed_c invT = ap_fixed_c(1) / ap_fixed_c(NUM_STEPS);
// 		output_stream.write(sum0 * invT);
// 		output_stream.write(sum1 * invT);

// // #ifndef __SYNTHESIS__
// //         std::cout << "After : out_stream size = " << output_stream.size() << std::endl;
// // #endif

//     }

// /* ================================================================
//  *  3.  LAYER OBJECTS  (unchanged)
//  * ================================================================ */
// private:
//     /* 1st block */
//     CONV1D_SD_FOLDED   <1, 12, 3, 1, CONV_IN_LENGTH1, ap_fixed<18, 10, AP_RND_CONV, AP_SAT>>       qcsnet2_cblk1_qconv1d_folded;
//     LIF1D_SD    <OUT_CH1, FEATURE_LENGTH1>           qcsnet2_cblk1_lif1d;
//     MaxPool1D_SD<2, 2, OUT_CH1, FEATURE_LENGTH1>     qcsnet2_cblk1_maxpool1d;

//     /* 2nd block */
//     CONV1D_SD_FOLDED   <12, 16, 3, 1, CONV_IN_LENGTH2, ap_fixed<24, 16, AP_RND_CONV, AP_SAT>>      qcsnet2_cblk2_qconv1d_folded;
//     LIF1D_SD    <OUT_CH2, FEATURE_LENGTH2>           qcsnet2_cblk2_lif1d;

//     /* 3rd block */
//     CONV1D_SD_FOLDED   <16, 20, 3, 1, CONV_IN_LENGTH3, ap_fixed<24, 16, AP_RND_CONV, AP_SAT>>      qcsnet2_cblk3_qconv1d_folded;
//     LIF1D_SD    <OUT_CH3, FEATURE_LENGTH3>           qcsnet2_cblk3_lif1d;
//     MaxPool1D_SD<2, 2, OUT_CH3, FEATURE_LENGTH3>     qcsnet2_cblk3_maxpool1d;

//     /* 4th block */
//     CONV1D_SD_FOLDED   <20, 24, 3, 1, CONV_IN_LENGTH4, ap_fixed<24, 16, AP_RND_CONV, AP_SAT>>      qcsnet2_cblk4_qconv1d_folded;
//     LIF1D_SD    <OUT_CH4, FEATURE_LENGTH4>           qcsnet2_cblk4_lif1d;
    
//     /* dense head */
//     Linear1D_SD <LINEAR_IN_SIZE, 2, ap_fixed<28, 20, AP_RND_CONV, AP_SAT>>                  qcsnet2_lblk1_qlinear1d;
//     LIF1D_SD    <2, 1>                               qcsnet2_lblk1_lif1d;
// };

// } // namespace hls4csnn1d_cblk_sd
// #endif /* NEURAL_NETWORK2_DEEPER_CBLK1_SD_H */
