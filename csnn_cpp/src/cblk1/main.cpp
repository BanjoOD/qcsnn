

// testbench_dump_layers.cpp
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <vector>
#include "hls_stream.h"
#include "ap_int.h"

#include "../../include/hls4csnn1d_sd/model4/constants4_sd.h"
#include "../../include/hls4csnn1d_sd/model4/filereader4.h"

// Include individual layer headers
#include "../../include/hls4csnn1d_sd/model4/cblk_sd/conv1d_sd.h"
#include "../../include/hls4csnn1d_sd/model4/cblk_sd/batchnorm1d_sd.h"
#include "../../include/hls4csnn1d_sd/model4/cblk_sd/lif1d_integer.h"
#include "../../include/hls4csnn1d_sd/model4/cblk_sd/maxpool1d_sd.h"
#include "../../include/hls4csnn1d_sd/model4/cblk_sd/linear1d_sd.h"
#include "../../include/hls4csnn1d_sd/model4/cblk_sd/nn4_cblk1_sd.h"
#include "../../include/hls4csnn1d_sd/model4/cblk_sd/modeleval4_sd.h"
#include "../../include/hls4csnn1d_sd/model4/cblk_sd/quantidentity1d_sd.h"



/* ================================================================
 *  Main Testbench
 * ================================================================ */

int main(int argc, char** argv) {
    std::cout << "\n=== QCSNN HLS Evaluation (4-Class + Softmax-Equivalent Weighting) ===\n\n";

    // -------------------------------------------------------------------------
    // 1. Resolve dataset folder
    // -------------------------------------------------------------------------
#ifndef __SYNTHESIS__
    std::string folderPath =
        "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_intra_patient_4class_180_center90_filtered/test";
#else
    std::string folderPath =
        "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_test/smallvhls";
#endif
    if (argc >= 2) {
        folderPath = argv[1];
    }

    // -------------------------------------------------------------------------
    // 2. Load data with FileReader
    // -------------------------------------------------------------------------
    FileReader reader;
    reader.loadData(folderPath);

    // Stream of full 1×180 ECG records
    hls::stream<array180_t> dataStreamInternal;
    reader.streamData(dataStreamInternal);

#ifndef __SYNTHESIS__
    const int NUM_SAMPLES_LOADED = dataStreamInternal.size();
#else
    const int NUM_SAMPLES_LOADED = NUM_SAMPLES;   // macro from constants_sd.h
#endif

    if (NUM_SAMPLES_LOADED <= 0) {
        std::cerr << "No data loaded!\n";
        return -1;
    }

    std::cout << "Loaded " << NUM_SAMPLES_LOADED << " ECG samples\n";

    // -------------------------------------------------------------------------
    // 3. Ground-truth labels: 4-class (0..3) for Normal, SVEB, VEB, F
    // -------------------------------------------------------------------------
    hls::stream<ap_int8_c> gtLabelStream;
    reader.streamLabel(gtLabelStream, /*binary=*/false);

    // -------------------------------------------------------------------------
    // 4. Instantiate model and evaluation helper
    // -------------------------------------------------------------------------
    hls4csnn1d_cblk_sd::NeuralNetwork4_Cblk1_sd<NUM_STEPS> model4;
    ModelEvaluation evaluator4;

    // -------------------------------------------------------------------------
    // 5. Multi-class confusion matrix
    //    cm[true][pred] for 4 classes: 0=Normal,1=SVEB,2=VEB,3=F
    // -------------------------------------------------------------------------
    const int N_CLASSES = 4;
    long cm[N_CLASSES][N_CLASSES];
    for (int i = 0; i < N_CLASSES; ++i) {
        for (int j = 0; j < N_CLASSES; ++j) {
            cm[i][j] = 0;
        }
    }

    // -------------------------------------------------------------------------
    // 6. SOFTMAX-EQUIVALENT LOGIT ADJUSTMENTS
    //    Python weights: Normal=0.628, SVEB=1.203, VEB=1.531, F=1.500
    //    Logit adjustment = -log(weight) scaled by 100
    //    Normal: -log(0.628) = +0.465 → +47
    //    SVEB:   -log(1.203) = -0.185 → -19
    //    VEB:    -log(1.531) = -0.426 → -43
    //    F:      -log(1.500) = -0.405 → -41
    // -------------------------------------------------------------------------
    const int LOGIT_ADJ_NORMAL = 47;   // Boost Normal (lower weight)
    const int LOGIT_ADJ_SVEB   = -19;  // Suppress SVEB (higher weight)
    const int LOGIT_ADJ_VEB    = -43;  // Suppress VEB (higher weight)
    const int LOGIT_ADJ_F      = -41;  // Suppress F (higher weight)

    std::cout << "Deployed weights (softmax-equivalent via logit adjustment):\n";
    std::cout << "  Normal: 0.628 (adj = +" << LOGIT_ADJ_NORMAL << ")\n";
    std::cout << "  SVEB:   1.203 (adj = " << LOGIT_ADJ_SVEB << ")\n";
    std::cout << "  VEB:    1.531 (adj = " << LOGIT_ADJ_VEB << ")\n";
    std::cout << "  F:      1.500 (adj = " << LOGIT_ADJ_F << ")\n\n";

    std::cout << "=== Per-sample predictions ===\n";

    // -------------------------------------------------------------------------
    // 7. Per-sample inference + confusion matrix update
    // -------------------------------------------------------------------------
    for (int n = 0; n < NUM_SAMPLES_LOADED; ++n) {
        // Flatten one ECG sample [array180_t → scalar stream]
        hls::stream<ap_int8_c> datastream;
        array180_t sample = dataStreamInternal.read();
        for (int i = 0; i < FIXED_LENGTH1; ++i) {
            datastream.write(sample[i]);
        }

        // Process through model (multi-class QCSNN)
        hls::stream<ap_int8_c> outStream;
        evaluator4.evaluate(model4, datastream, outStream);

        // Read 4 outputs from the model (averaged spike counts per class)
        ap_int8_c y0 = outStream.read();   // class 0: Normal
        ap_int8_c y1 = outStream.read();   // class 1: SVEB
        ap_int8_c y2 = outStream.read();   // class 2: VEB
        ap_int8_c y3 = outStream.read();   // class 3: F

        // ========== SOFTMAX-EQUIVALENT WEIGHTING VIA LOGIT ADJUSTMENT ==========
        // Adjusted logit = raw_logit * 100 + adjustment
        ap_int<16> adj0 = (ap_int<16>)y0 * 100 + LOGIT_ADJ_NORMAL;
        ap_int<16> adj1 = (ap_int<16>)y1 * 100 + LOGIT_ADJ_SVEB;
        ap_int<16> adj2 = (ap_int<16>)y2 * 100 + LOGIT_ADJ_VEB;
        ap_int<16> adj3 = (ap_int<16>)y3 * 100 + LOGIT_ADJ_F;

        // Argmax over adjusted logits
        int pred_class = 0;
        ap_int<16> best = adj0;

        if (adj1 > best) {
            best = adj1;
            pred_class = 1;
        }
        if (adj2 > best) {
            best = adj2;
            pred_class = 2;
        }
        if (adj3 > best) {
            best = adj3;
            pred_class = 3;
        }
        // =======================================================================

        // True label (0..3)
        int y_true = (int)gtLabelStream.read();
        if (y_true < 0 || y_true >= N_CLASSES) {
            std::cerr << "Warning: label out of range (" << y_true << ") at sample " << n << "\n";
            continue;
        }

        cm[y_true][pred_class]++;

#ifndef __SYNTHESIS__
        if (n < 20) {
            std::cout << "Sample " << n
                      << "  y_true=" << y_true
                      << "  pred=" << pred_class
                      << "  raw=[" << (int)y0 << "," << (int)y1 
                      << "," << (int)y2 << "," << (int)y3
                      << "]  adj=[" << (int)adj0 << "," << (int)adj1
                      << "," << (int)adj2 << "," << (int)adj3 << "]\n";
        }
#endif
    }

    // -------------------------------------------------------------------------
    // 8. Print confusion matrix
    // -------------------------------------------------------------------------
    std::cout << "\n=== Confusion Matrix [true x pred] ===\n";
    std::cout << "      pred:  0      1      2      3\n";
    for (int i = 0; i < N_CLASSES; ++i) {
        std::cout << "true " << i << ": ";
        for (int j = 0; j < N_CLASSES; ++j) {
            std::cout << std::setw(6) << cm[i][j] << " ";
        }
        std::cout << "\n";
    }

    // -------------------------------------------------------------------------
    // 9. Per-class metrics + overall accuracy
    // -------------------------------------------------------------------------
    const double eps = 1e-12;
    long total_correct = 0;
    long total_samples = 0;

    for (int i = 0; i < N_CLASSES; ++i) {
        for (int j = 0; j < N_CLASSES; ++j) {
            total_samples += cm[i][j];
            if (i == j) {
                total_correct += cm[i][j];
            }
        }
    }

    static const char* CLASS_NAMES[N_CLASSES] = {
        "Normal (N)",
        "SVEB (S)",
        "VEB (V)",
        "F (F)"
    };

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n=== Per-class metrics (softmax-equivalent weighting) ===\n";

    for (int c = 0; c < N_CLASSES; ++c) {
        long TP = cm[c][c];
        long FN = 0;
        long FP = 0;
        long TN = 0;

        for (int i = 0; i < N_CLASSES; ++i) {
            for (int j = 0; j < N_CLASSES; ++j) {
                long val = cm[i][j];
                if (i == c && j != c) {
                    FN += val;
                } else if (i != c && j == c) {
                    FP += val;
                } else if (i != c && j != c) {
                    TN += val;
                }
            }
        }

        double precision = TP / (double)(TP + FP + eps);
        double recall    = TP / (double)(TP + FN + eps);
        double f1        = (2.0 * precision * recall) / (precision + recall + eps);
        double accuracy  = (TP + TN) / (double)(TP + TN + FP + FN + eps);

        std::cout << "\nClass " << c << " - " << CLASS_NAMES[c] << "\n"
                  << "  Accuracy : " << accuracy  << "\n"
                  << "  Precision: " << precision << "\n"
                  << "  Recall   : " << recall    << "\n"
                  << "  F1       : " << f1        << "\n";
    }

    double acc_global = total_correct / (double)(total_samples + eps);
    std::cout << "\n=== Overall multi-class accuracy ===\n"
              << "Accuracy : " << acc_global << "\n";

    std::cout << "\n=== Complete ===\n";
    return 0;
}



// ##############################################################################################   /////

// // testbench_dump_layers.cpp
// #include <iostream>
// #include <fstream>
// #include <string>
// #include <iomanip>
// #include <vector>
// #include "hls_stream.h"
// #include "ap_int.h"

// #include "../../include/hls4csnn1d_sd/model2/constants_sd.h"
// #include "../../include/hls4csnn1d_sd/model2/filereader.h"
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/nn2_cblk1_sd.h"

// // Include individual layer headers
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/conv1d_sd.h"
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/batchnorm1d_sd.h"
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/lif1d_integer.h"
// // #include "../../include/hls4csnn1d_sd/model2/cblk_sd/lif1d_float.h"
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/maxpool1d_sd.h"
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/linear1d_sd.h"
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/nn2_cblk1_sd.h"
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/modeleval_sd.h"
// // Add include
// #include "../../include/hls4csnn1d_sd/model2/cblk_sd/quantidentity1d_sd.h"


// /* ================================================================
//  *  Main Testbench
//  * ================================================================ */
// int main(int argc, char** argv) {
//     std::cout << "\n=== QCSNN HLS Evaluation (NeuralNetwork2_Cblk1_sd + ModelEvaluation) ===\n\n";

//     // -------------------------------------------------------------------------
//     // 1. Resolve dataset folder
//     // -------------------------------------------------------------------------
//     std::string folderPath =
//         "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_intra_patient_4class_180_center90_filtered/test";
//     if (argc >= 2) {
//         folderPath = argv[1];
//     }

//     // -------------------------------------------------------------------------
//     // 2. Load data with FileReader (same pattern as before)
//     // -------------------------------------------------------------------------
//     FileReader reader;
//     reader.loadData(folderPath);

//     // Stream of full 1×180 records (as in your previous main)
//     hls::stream<array180_t> dataStreamInternal;
//     reader.streamData(dataStreamInternal);

// #ifndef __SYNTHESIS__
//     const int NUM_SAMPLES_LOADED = dataStreamInternal.size();
// #else
//     const int NUM_SAMPLES_LOADED = NUM_SAMPLES;   // macro from constants_sd.h
// #endif

//     if (NUM_SAMPLES_LOADED <= 0) {
//         std::cerr << "No data loaded!\n";
//         return -1;
//     }

//     std::cout << "Loaded " << NUM_SAMPLES_LOADED << " ECG samples\n";

//     // Ground-truth labels (binary: 0 normal, 1 abnormal)
//     hls::stream<ap_int8_c> gtLabelStream;
//     reader.streamLabel(gtLabelStream, /*binary=*/true);

//     // -------------------------------------------------------------------------
//     // 3. Flatten array180_t stream → scalar ap_int8_c stream for ModelEvaluation
//     // -------------------------------------------------------------------------
//     hls4csnn1d_cblk_sd::NeuralNetwork2_Cblk1_sd<NUM_STEPS> model2;
//     ModelEvaluation evaluator;

//     // -------------------------------------------------------------------------
//     // 4. Threshold configuration (MODIFIED)
//     // -------------------------------------------------------------------------
//     const int THRESHOLD_NUM = 32;   // P(abnormal) >= 0.32
//     const int THRESHOLD_DEN = 68;   // 1.0 - 0.32 = 0.68
//     const float THRESHOLD_VALUE = (float)THRESHOLD_NUM / (THRESHOLD_NUM + THRESHOLD_DEN);

//     // -------------------------------------------------------------------------
//     // 5. Compute per-class metrics from outputs vs. ground truth
//     // -------------------------------------------------------------------------
//     long TP0 = 0, TN0 = 0, FP0 = 0, FN0 = 0;
//     long TP1 = 0, TN1 = 0, FP1 = 0, FN1 = 0;

//     std::cout << "\n=== Per-sample predictions (threshold=" << THRESHOLD_VALUE << ") ===\n";

//     for (int n = 0; n < NUM_SAMPLES_LOADED; ++n) {
//         // Flatten one sample
//         hls::stream<ap_int8_c> datastream;
//         array180_t sample = dataStreamInternal.read();
//         for (int i = 0; i < FIXED_LENGTH1; ++i) {
//             datastream.write(sample[i]);
//         }
        
//         // Process through model
//         hls::stream<ap_int8_c> outStream;
//         evaluator.evaluate(model2, datastream, outStream);
        
//         // Get predictions (averaged spike counts as int8)
//         ap_int8_c y0 = outStream.read();   // class 0 score
//         ap_int8_c y1 = outStream.read();   // class 1 score

//         // ============ MODIFIED: Apply threshold = 0.32 ============
//         // Original: int pred_class = ( (int)y1 > (int)y0 ) ? 1 : 0;
//         // New: y1/(y0+y1) >= 0.32  →  y1 * 68 >= y0 * 32
//         int pred_class;
//         if ((int)y1 * THRESHOLD_DEN >= (int)y0 * THRESHOLD_NUM) {
//             pred_class = 1;  // abnormal
//         } else {
//             pred_class = 0;  // normal
//         }
//         // ==========================================================

//         int y_true = (int)gtLabelStream.read();

//         // Update confusion matrices
//         if (pred_class == 1 && y_true == 1) { ++TP1; ++TN0; }
//         else if (pred_class == 0 && y_true == 0) { ++TN1; ++TP0; }
//         else if (pred_class == 1 && y_true == 0) { ++FP1; ++FN0; }
//         else if (pred_class == 0 && y_true == 1) { ++FN1; ++FP0; }
//     }

//     // -------------------------------------------------------------------------
//     // 6. Final metrics (normal vs abnormal), same definitions as before
//     // -------------------------------------------------------------------------
//     const double eps = 1e-12;
//     const double TP1d = double(TP1), TN1d = double(TN1), FP1d = double(FP1), FN1d = double(FN1);
//     const double TP0d = double(TP0), TN0d = double(TN0), FP0d = double(FP0), FN0d = double(FN0);

//     // Abnormal class (1) as positive
//     const double precision1 = TP1d / (TP1d + FP1d + eps);
//     const double recall1    = TP1d / (TP1d + FN1d + eps);
//     const double f11        = (2.0 * precision1 * recall1) / (precision1 + recall1 + eps);
//     const double accuracy1  = (TP1d + TN1d) / (TP1d + TN1d + FP1d + FN1d + eps);

//     // Normal class (0) as positive
//     const double precision0 = TP0d / (TP0d + FP0d + eps);
//     const double recall0    = TP0d / (TP0d + FN0d + eps);
//     const double f10        = (2.0 * precision0 * recall0) / (precision0 + recall0 + eps);
//     const double accuracy0  = (TP0d + TN0d) / (TP0d + TN0d + FP0d + FN0d + eps);

//     // Global accuracy (identical to accuracy0/accuracy1 in binary case)
//     const double correct = TP1d + TN1d;
//     const double total   = TP1d + TN1d + FP1d + FN1d + eps;
//     const double acc_global = correct / total;

//     std::cout << std::fixed << std::setprecision(4)
//               << "\n=== Metrics for normal class (label 0) ===\n"
//               << "Accuracy : " << accuracy0  << "\n"
//               << "Precision: " << precision0 << "\n"
//               << "Recall   : " << recall0    << "\n"
//               << "F1       : " << f10        << "\n\n";

//     std::cout << std::fixed << std::setprecision(4)
//               << "=== Metrics for abnormal class (label 1) ===\n"
//               << "Accuracy : " << accuracy1  << "\n"
//               << "Precision: " << precision1 << "\n"
//               << "Recall   : " << recall1    << "\n"
//               << "F1       : " << f11        << "\n\n";

//     std::cout << std::fixed << std::setprecision(4)
//               << "=== Overall binary accuracy ===\n"
//               << "Accuracy : " << acc_global << "\n";

//     std::cout << "\n=== Complete ===\n";
//     return 0;
// }

