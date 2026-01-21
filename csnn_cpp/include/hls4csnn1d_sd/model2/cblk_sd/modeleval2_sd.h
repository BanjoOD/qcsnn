
// #ifndef MODEL_EVAL2_H
// #define MODEL_EVAL2_H

// #include "../constants_sd.h"
// #include "hls_stream.h"
// #include "ap_fixed.h"
// #include "nn2_deeper_cblk1_sd.h"


// class ModelEvaluation2 {
//     public:
//         // void evaluate(hls4csnn1d_bm::NeuralNetwork_bm& model4, hls::stream<input_data_type>& datastream, hls::stream<ap_fixed_c>& labelstream) {
//         //     int sample_count = 0;
//         //     int datastream_read_count = 0;

//         //     // Process input data as long as there is input and room for output
//         //     while (!datastream.empty()) {
//         //         #pragma HLS PIPELINE off
    
//         //         // Read one sample from the input stream
//         //         input_data_type input_array = datastream.read();
//         //         datastream_read_count++;
    
//         //         // Prepare a stream for model4
//         //         hls::stream<ap_fixed_c> input_stream4;
//         //         #pragma HLS STREAM variable=input_stream4 depth=FIXED_LENGTH
    
//         //         // Duplicate input data into input_stream4
//         //         for (int i = 0; i < FIXED_LENGTH; i++) {
//         //             #pragma HLS PIPELINE
//         //             ap_fixed_c data = input_array[i];
//         //             if (!input_stream4.full()) {
//         //                 input_stream4.write(data);
//         //             }
//         //         }
    
//         //         // Forward pass through model4
//         //         hls::stream<ap_fixed_c> out_stream4;
//         //         #pragma HLS STREAM variable=out_stream4 depth=NUM_CLASSES_4
//         //         model4.forward(input_stream4, out_stream4);
    
//         //         // Read the model outputs
//         //         ap_fixed_c out4[NUM_CLASSES_4];
//         //         for (int i = 0; i < NUM_CLASSES_4; i++) {
//         //             #pragma HLS PIPELINE
//         //             if (!out_stream4.empty()) {
//         //                 out4[i] = out_stream4.read();
//         //             }
//         //         }
    
//         //         // Compute softmax on the model outputs
//         //         ap_fixed_c softmax_out4[NUM_CLASSES_4];
//         //         #pragma HLS ARRAY_PARTITION variable=softmax_out4 complete
//         //         softmax<NUM_CLASSES_4>(out4, softmax_out4);
    
//         //         // Determine the predicted class
//         //         int pred4 = 0;
//         //         ap_fixed_c max_val4 = softmax_out4[0];
//         //         for (int i = 1; i < NUM_CLASSES_4; i++) {
//         //             #pragma HLS PIPELINE
//         //             if (softmax_out4[i] > max_val4) {
//         //                 max_val4 = softmax_out4[i];
//         //                 pred4 = i;
//         //             }
//         //         }
    
//         //         // Stream the predicted label into the output stream.
//         //         // Casting the integer prediction to the ap_fixed_c type.
//         //         if(!labelstream.full()) {
//         //             labelstream.write(static_cast<ap_fixed_c>(pred4));
//         //         }
                
//         //     }

//         //     std::cout << "data stream count: " << datastream_read_count << std::endl;
//         // }


//         // void evaluate(hls4csnn1d_bm::NeuralNetwork2_bm& model2, hls::stream<input_data_type>& datastream, hls::stream<ap_fixed_c>& labelstream) {
//         //     int sample_count = 0;
//         //     int datastream_read_count = 0;

//         //     // Process input data as long as there is input and room for output
//         //     while (!datastream.empty()) {
//         //         #pragma HLS PIPELINE off
    
//         //         // Read one sample from the input stream
//         //         input_data_type input_array = datastream.read();
//         //         datastream_read_count++;
    
//         //         // Prepare a stream for model4
//         //         hls::stream<ap_fixed_c> input_stream2;
//         //         #pragma HLS STREAM variable=input_stream2 depth=FIXED_LENGTH
    
//         //         // Duplicate input data into input_stream4
//         //         for (int i = 0; i < FIXED_LENGTH; i++) {
//         //             #pragma HLS PIPELINE
//         //             ap_fixed_c data = input_array[i];
//         //             if (!input_stream2.full()) {
//         //                 input_stream2.write(data);
//         //             }
//         //         }
    
//         //         // Forward pass through model4
//         //         hls::stream<ap_fixed_c> out_stream2;
//         //         #pragma HLS STREAM variable=out_stream2 depth=NUM_CLASSES_2
//         //         model2.forward(input_stream2, out_stream2);
    
//         //         // Read the model outputs
//         //         ap_fixed_c out2[NUM_CLASSES_2];
//         //         for (int i = 0; i < NUM_CLASSES_2; i++) {
//         //             #pragma HLS PIPELINE
//         //             if (!out_stream2.empty()) {
//         //                 out2[i] = out_stream2.read();
//         //             }
//         //         }
    
//         //         // Compute softmax on the model outputs
//         //         ap_fixed_c softmax_out2[NUM_CLASSES_2];
//         //         #pragma HLS ARRAY_PARTITION variable=softmax_out2 complete
//         //         softmax<NUM_CLASSES_2>(out2, softmax_out2);
    
//         //         // Determine the predicted class
//         //         int pred2 = 0;
//         //         ap_fixed_c max_val2 = softmax_out2[0];
//         //         for (int i = 1; i < NUM_CLASSES_2; i++) {
//         //             #pragma HLS PIPELINE
//         //             if (softmax_out2[i] > max_val2) {
//         //                 max_val2 = softmax_out2[i];
//         //                 pred2 = i;
//         //             }
//         //         }
    
//         //         // Stream the predicted label into the output stream.
//         //         // Casting the integer prediction to the ap_fixed_c type.
//         //         if(!labelstream.full()) {
//         //             labelstream.write(static_cast<ap_fixed_c>(pred2));
//         //         }
                
//         //     }

//         //     std::cout << "data stream count: " << datastream_read_count << std::endl;
//         // }


//         void evaluate(hls4csnn1d_cblk_sd::NeuralNetwork2_Cblk1_sd& model2, 
//                       hls::stream<input180_data_type>& datastream, 
//                       hls::stream<input2_data_type>& labelstream) {
//             int datastream_read_count = 0;

//             // Process all input samples from the data stream
//             while (!datastream.empty()) {
//                 #pragma HLS PIPELINE off

//                 // Read one sample from the input stream
//                 input180_data_type input_array = datastream.read();
//                 datastream_read_count++;

//                 // Prepare a stream for the model input
//                 hls::stream<ap_fixed_c> input_stream;
//                 #pragma HLS STREAM variable=input_stream depth=FIXED_LENGTH1

//                 // Transfer input array data into the stream
//                 for (int i = 0; i < FIXED_LENGTH1; i++) {
//                     #pragma HLS PIPELINE
//                     input_stream.write(input_array[i]);
//                 }

//                 // Forward pass: process the input stream to get the model output stream
//                 hls::stream<ap_fixed_c> output_stream;
//                 #pragma HLS STREAM variable=output_stream depth=FIXED_LENGTH8
// #ifndef __SYNTHESIS__
//         std::cout << "ModelEval Before : out_stream size = " << output_stream.size() << std::endl;
// #endif
//                 model2.forward(input_stream, output_stream);

// #ifndef __SYNTHESIS__
//         std::cout << "ModelEval After : out_stream size = " << output_stream.size() << std::endl;
// #endif
//                 while (!output_stream.empty()) {
               
//                     // Collect the model's output into an output array
//                     input2_data_type output_array;
//                     for (int i = 0; i < FIXED_LENGTH8; i++) {
//                         #pragma HLS PIPELINE
//                         output_array[i] = output_stream.read();
//                     }

//                     // Write the output array to the label stream if there's room
//                     if (!labelstream.full()) {
//                     labelstream.write(output_array);
//                     }
//                 }

//             }

// //             std::cout << "Data stream read count: " << datastream_read_count << std::endl;
//         }



        

       

        
//  };
    
// #endif // MODEL_EVAL_H
