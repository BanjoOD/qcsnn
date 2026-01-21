#ifndef LINEAR1D_SD_H
#define LINEAR1D_SD_H

#include <hls_stream.h>
#include <ap_int.h>
#include "../constants_sd.h"
#ifndef __SYNTHESIS__
#include <iostream>
#endif

namespace hls4csnn1d_cblk_sd {


/**
 * Fully-connected layer with quantized integer operations
 * Template parameters control optional bias and asymmetric quantization
 */
template<int INPUT_SIZE, int OUTPUT_SIZE,
         bool USE_BIAS = false,
         bool USE_ASYMMETRIC = false>
class Linear1D_SD {
public:
    Linear1D_SD() {}

    void forward(hls::stream<ap_int8_c>& in_stream,
                 hls::stream<ap_int8_c>& out_stream,
                 const ap_int8_c weights[OUTPUT_SIZE][INPUT_SIZE],
                 const ap_int<32> scale_multiplier[OUTPUT_SIZE],
                 const int right_shift[OUTPUT_SIZE],
                 const acc32_t bias[OUTPUT_SIZE],
                 const ap_int<8> input_zero_point,
                 const acc32_t weight_sum[OUTPUT_SIZE]) {
    #pragma HLS ARRAY_PARTITION variable=weights complete dim=1
    #pragma HLS ARRAY_PARTITION variable=weights complete dim=2
    #pragma HLS ARRAY_PARTITION variable=scale_multiplier complete dim=1
    #pragma HLS ARRAY_PARTITION variable=right_shift complete dim=1
    #pragma HLS INTERFACE mode=s_axilite port=scale_multiplier
    #pragma HLS INTERFACE mode=s_axilite port=right_shift
    #pragma HLS INTERFACE mode=s_axilite port=bias
    #pragma HLS INTERFACE mode=s_axilite port=input_zero_point
    #pragma HLS INTERFACE mode=s_axilite port=weight_sum

        // 1) Read input vector
        ap_int8_c in_vec[INPUT_SIZE];
    #pragma HLS ARRAY_PARTITION variable=in_vec complete

        READ_IN:
        for (int i = 0; i < INPUT_SIZE; ++i) {
        #pragma HLS PIPELINE II=1
            in_vec[i] = in_stream.read();
        }

#ifndef __SYNTHESIS__
        int lin_sat_hi = 0, lin_sat_lo = 0;
#endif

        // 2) Compute output: MAC + requantization
        DOT_OUT:
        for (int o = 0; o < OUTPUT_SIZE; ++o) {
        #pragma HLS PIPELINE II=1
            acc32_t acc = 0;

            // Integer MAC (int8 × int8 → int32)
            MUL_ADD:
            for (int i = 0; i < INPUT_SIZE; ++i) {
            #pragma HLS UNROLL
                acc += (acc32_t)weights[o][i] * (acc32_t)in_vec[i];
            }

            // Optional: Add bias (already in accumulator domain)
            if (USE_BIAS) {
                acc += bias[o];
            }

            // Optional: Asymmetric quantization correction
            if (USE_ASYMMETRIC) {
                acc -= (acc32_t)input_zero_point * weight_sum[o];
            }

            // Requantize to next layer's int8 scale
            ap_int8_c y_q = requantize(acc, scale_multiplier[o], right_shift[o]);

#ifndef __SYNTHESIS__
            if (y_q == ap_int8_c(127))  ++lin_sat_hi;
            if (y_q == ap_int8_c(-128)) ++lin_sat_lo;
#endif
            out_stream.write(y_q);
        }

#ifndef __SYNTHESIS__
        static int lin_once = 0;
        if (lin_once++ < 1) {
            std::cout << "[Linear1D_SD] sat_hi=" << lin_sat_hi
                      << " sat_lo=" << lin_sat_lo << "\n";
        }
#endif
    }
};

} // namespace hls4csnn1d_cblk_sd
#endif // LINEAR1D_SD_H 





// #ifndef LINEAR1D_SD_H
// #define LINEAR1D_SD_H

// #include <hls_stream.h>
// #include <ap_fixed.h>
// #ifndef __SYNTHESIS__
// #include <fstream>
// #include <iostream>
// #endif

// #include "../constants_sd.h"  // must define: using ap_fixed_c = ap_fixed<8,4,AP_CONV,AP_SAT>

// namespace hls4csnn1d_cblk_sd {

// // Wider accumulator for MACs: conservative <16,8>. Adjust if you have tighter bounds.
// // using lin_acc_t = ap_fixed<32, 16, AP_RND_CONV, AP_SAT>;

// /**
//  * Fully-connected layer: OUT = W × IN (+ optional bias)
//  *  – INPUT_SIZE, OUTPUT_SIZE are compile-time
//  *  – Weights are passed as a const 2D array (partitioned on dim=2)
//  */
// template<int INPUT_SIZE, int OUTPUT_SIZE, typename lin_acc_t>
// class Linear1D_SD {
// public:
//     Linear1D_SD() {}

//     void forward(
//         hls::stream<ap_fixed_c>& in,
//         hls::stream<ap_fixed_c>& out,
//         const ap_fixed_c weights[OUTPUT_SIZE][INPUT_SIZE],
//         const ap_fixed_c *bias = nullptr)   // optional bias
//     {
//     #pragma HLS ARRAY_PARTITION variable=weights complete dim=2

//         // 1) Read input vector
//         ap_fixed_c in_vec[INPUT_SIZE];
//     #pragma HLS ARRAY_PARTITION variable=in_vec complete

//         READ_IN:
//         for (int i = 0; i < INPUT_SIZE; ++i) {
//         #pragma HLS PIPELINE II=1
//             in_vec[i] = in.read();
//         }

// #ifndef __SYNTHESIS__
//         int lin_sat_hi = 0, lin_sat_lo = 0;
// #endif

//         // 2) Compute output = W * x (+ b) with widened accumulator
//         DOT_OUT:
//         for (int o = 0; o < OUTPUT_SIZE; ++o) {
//         #pragma HLS PIPELINE II=1
//             lin_acc_t acc = bias ? (lin_acc_t)bias[o] : (lin_acc_t)0;

//             MUL_ADD:
//             for (int i = 0; i < INPUT_SIZE; ++i) {
//             #pragma HLS UNROLL
//                 acc += (lin_acc_t)weights[o][i] * (lin_acc_t)in_vec[i];
//             }
//             ap_fixed_c y = ap_fixed_c(acc);      // final cast to stream type
// #ifndef __SYNTHESIS__
//             if (y == ap_fixed_c(1.984375)) ++lin_sat_hi;
//             if (y == ap_fixed_c(-2.0))   ++lin_sat_lo;
// #endif

//             // Single cast back to layer format (convergent + saturate)
//             out.write((ap_fixed_c)acc);
//         }
// #ifndef __SYNTHESIS__
//             static int lin_once = 0;
//             if (lin_once++ < 1) {
//                 std::cout << "[Linear] sat_hi=" << lin_sat_hi
//                     << " sat_lo=" << lin_sat_lo << "\n";
//              }
// #endif
//     }
// };

// } // namespace hls4csnn1d_cblk_sd
// #endif // LINEAR1D_SD_H



// #ifndef LINEAR1D_SD_H
// #define LINEAR1D_SD_H

// #include <hls_stream.h>
// #include <ap_fixed.h>
// #ifndef __SYNTHESIS__
// #include <fstream>
// #include <iostream>
// #endif

// #include "../constants_sd.h"      // brings in ap_fixed_c and your WEIGHT file macros
// namespace hls4csnn1d_cblk_sd {

// /**
//  * Fully‑connected layer: OUT = W × IN
//  *  – INPUT_SIZE, OUTPUT_SIZE are compile‑time
//  *  – Weights loaded once in C‑sim from ASCII file LINEAR_WFILE
//  */
// template<int INPUT_SIZE, int OUTPUT_SIZE>
// class Linear1D_SD {
// public:
//     Linear1D_SD() {}

//     void forward(
//     hls::stream<ap_fixed_c>& in,
//     hls::stream<ap_fixed_c>& out,
//     const ap_fixed_c weights[OUTPUT_SIZE][INPUT_SIZE]) {
// #pragma HLS ARRAY_PARTITION variable=weights complete dim=2

//     // ========== DEBUG ADDITION START ==========
// // #ifndef __SYNTHESIS__
// //     static int call_count = 0;
// //     call_count++;
    
// //     if (call_count <= 2) {  // Debug first 2 calls
// //         std::cout << "\n=== Linear Debug (call " << call_count << ") ===\n";
// //         std::cout << "Input size: " << INPUT_SIZE << ", Output size: " << OUTPUT_SIZE << "\n";
        
// //         // Print first few weights
// //         std::cout << "Sample weights:\n";
// //         for (int o = 0; o < OUTPUT_SIZE && o < 2; ++o) {
// //             std::cout << "Output " << o << " weights (first 10): ";
// //             for (int i = 0; i < 10 && i < INPUT_SIZE; ++i) {
// //                 std::cout << weights[o][i] << " ";
// //             }
// //             std::cout << "\n";
// //         }
// //     }
// // #endif
//     // ========== DEBUG ADDITION END ==========
    
//     // Read input vector
//     ap_fixed_c in_vec[INPUT_SIZE];
// #pragma HLS ARRAY_PARTITION variable=in_vec complete
//     READ_IN:
//     for (int i = 0; i < INPUT_SIZE; ++i) {
//     #pragma HLS PIPELINE II=1
//         in_vec[i] = in.read();
        
//         // ========== DEBUG ADDITION START ==========
// // #ifndef __SYNTHESIS__
// //         if (call_count <= 2 && i < 20) {  // Print first 20 inputs
// //             std::cout << "Linear input[" << i << "]: " << in_vec[i] << "\n";
// //         }
        
// //         // Count non-zero inputs
// //         static int nonzero_count = 0;
// //         if (call_count <= 2) {
// //             if (in_vec[i] != 0) nonzero_count++;
// //             if (i == INPUT_SIZE - 1) {
// //                 std::cout << "Non-zero inputs: " << nonzero_count << "/" << INPUT_SIZE << "\n";
// //                 nonzero_count = 0;  // Reset for next call
// //             }
// //         }
// // #endif
//         // ========== DEBUG ADDITION END ==========
//     }
    
//     // Compute output = weights × input vector
//     DOT_OUT:
//     for (int o = 0; o < OUTPUT_SIZE; ++o) {
//     #pragma HLS PIPELINE II=1
//         ap_fixed_c acc = 0;
//         MUL_ADD:
//         for (int i = 0; i < INPUT_SIZE; ++i) {
//         #pragma HLS UNROLL
//             acc += weights[o][i] * in_vec[i];
            
//             // ========== DEBUG ADDITION START ==========
// // #ifndef __SYNTHESIS__
// //             // Debug accumulation for first few outputs
// //             if (call_count <= 2 && o < 2 && i < 5) {
// //                 std::cout << "Output[" << o << "] += " << weights[o][i] 
// //                           << " * " << in_vec[i] << " = " 
// //                           << (weights[o][i] * in_vec[i]) << ", acc=" << acc << "\n";
// //             }
// // #endif
//             // ========== DEBUG ADDITION END ==========
//         }
        
//         // ========== DEBUG ADDITION START ==========
// // #ifndef __SYNTHESIS__
// //         if (call_count <= 2) {
// //             std::cout << "Linear output[" << o << "]: " << acc << "\n";
// //             if (acc == ap_fixed_c(1.98438)) {
// //                 std::cout << "WARNING: Linear output saturated at 1.98438!\n";
// //             }
// //         }
// // #endif
//         // ========== DEBUG ADDITION END ==========
        
//         out.write(acc);
//     }
// }

    



// // void forward(
//     //     hls::stream<ap_fixed_c>& in,
//     //     hls::stream<ap_fixed_c>& out,
//     //     const ap_fixed_c weights[OUTPUT_SIZE][INPUT_SIZE]) {
//     // #pragma HLS ARRAY_PARTITION variable=weights complete dim=2     
//     //     // Read input vector
//     //     ap_fixed_c in_vec[INPUT_SIZE];
//     // #pragma HLS ARRAY_PARTITION variable=in_vec complete
//     //     READ_IN:
//     //     for (int i = 0; i < INPUT_SIZE; ++i) {
//     //     #pragma HLS PIPELINE II=1
//     //         in_vec[i] = in.read();
//     //     }

//     //     // Compute output = weights × input vector
//     //     DOT_OUT:
//     //     for (int o = 0; o < OUTPUT_SIZE; ++o) {
//     //     #pragma HLS PIPELINE II=1
//     //         ap_fixed_c acc = 0;
//     //         MUL_ADD:
//     //         for (int i = 0; i < INPUT_SIZE; ++i) {
//     //         #pragma HLS UNROLL
//     //             acc += weights[o][i] * in_vec[i];
//     //         }
//     //         out.write(acc);
//     //     }
//     // }

// };

// } // namespace hls4csnn1d_cblk_sd
// #endif // LINEAR1D_SD_H
