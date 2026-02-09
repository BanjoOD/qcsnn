# QCSNN: Quantized Convolution Spiking Neural Network

This repository implements an end-to-end **Quantized Convolution Spiking Neural Network (QCSNN)** for ECG arrhythmia classification and accelerates it on a **PYNQ-Z2 FPGA**.

The project is split into two major parts:

1. **Model Development** (PyTorch + snnTorch + Brevitas)  
2. **Model Acceleration** (C++ / HLS → Vitis → Vivado → PYNQ-Z2)

The PyTorch models live in the Jupyter notebooks, while the C++ / HLS implementation (combined 2-stage QCSNN) lives in the `csnn_cpp` folder.

---

## 1. Repository Structure (high level)

- `notebooks/`
  - Stage-1 (binary) QCSNN training and weight export  
  - Stage-2 (4-class) QCSNN training and weight export  
  - Optional combined QCSNN evaluation (Python reference, for comparison with C++)
- `csnn_cpp/`
  - C++ implementation of the **combined 2-stage QCSNN**  
  - `weights_sd/` – directory where exported PyTorch weights are copied for C++ / HLS  
  - HLS-friendly layer implementations (e.g., Conv1D, LIF, etc.)

(Adjust folder names to your actual layout.)

---

## 2. End-to-End Workflow

### 2.1. Model Development (PyTorch)

> Goal: train **two QCSNN models** in PyTorch (Stage-1 binary, Stage-2 multi-class), export their quantized weights, and optionally evaluate a combined Python cascade.

1. **Run Stage-1 (binary) notebook**
   - Open the Stage-1 QCSNN notebook in `notebooks/`.
   - Train the binary model (Normal vs Abnormal).
   - Run the cells that **export the trained, quantized weights**.

2. **Run Stage-2 (4-class) notebook**
   - Open the Stage-2 QCSNN notebook.
   - Train the 4-class model (e.g., N / SVEB / VEB / F).
   - Run the cells that **export the trained, quantized weights**.

3. **[Optional] Run the combined-stage notebook**
   - Open the combined/cascaded notebook.
   - Evaluate the 2-stage QCSNN in PyTorch to obtain reference metrics:
     - Accuracy, macro-F1
     - Per-class metrics
     - Confusion matrices  
   - These are later used to compare against the C++ / FPGA implementation.

4. **Copy exported weights into the C++ project**
   - Take the exported weights (e.g., `.npy`, `.txt`, `.json`—depending on your export format).
   - Copy / move them into `csnn_cpp/weights_sd/` (or the weight folder expected by your C++ code).

At this point, the **C++ / HLS project has all the trained weights** and you can switch to model acceleration.

---

### 2.2. Model Acceleration (C++ / HLS → Vivado → PYNQ-Z2)

> Goal: use the C++ implementation of the **combined QCSNN** and accelerate it via Vitis + Vivado, targeting a PYNQ-Z2 FPGA.

#### A. Vitis Unified IDE (HLS Component)

1. **Create a Component in Vitis Unified IDE**
   - Create a new **HLS Component** (C/C++ kernel) for the QCSNN.
   - Add or point to the C++ sources in `csnn_cpp/` (top-level spiking network, layers, weight loading, etc.).

2. **Run the HLS flow**
   - Configure the **top function** (QCSNN inference entry) and IO interfaces (e.g., AXI4-Stream / AXI4-Lite).
   - Run:
     - **C Simulation** – functional check with test data.
     - **C Synthesis** – generate RTL from C++.
     - *(Optional but recommended)* **C/RTL Co-simulation** – verify RTL vs C++.

3. **Package / export for RTL integration**
   - For the “classic” Vivado block design flow (used here):
     - **Package/export the synthesized component as an IP** and import it in Vivado.
   - For a fully Vitis-centric flow:
     - You can also run **Implementation** and **Package** in Vitis to generate a **hardware handoff (.xsa)** for use in system projects.
   - In this project, **implementation and bitstream are run in Vivado**, with Vitis used primarily for HLS and IP packaging.

#### B. Vivado Block Design (IP Integration)

1. **Create a Vivado block design** and add:
   - **Zynq7 Processing System** (PYNQ-Z2 PS).
   - **Custom QCSNN IP** (exported from Vitis HLS).
   - **AXI DMA**  
     - For streaming ECG data in and predictions out.
   - **AXI Interconnect**  
     - For routing AXI-Lite control interfaces between PS and custom IP.
   - **AXI Memory Interconnect**  
     - For connecting AXI DMA to DDR memory via HP ports on the PS.
   - **Processor System Reset**  
     - To generate reset signals for AXI and custom IP from PS clocks.
   - **xlconstant / ilconstant IP**  
     - For tying unused or constant-control signals to fixed logic values (e.g., enable, mode bits).

2. **Hook up the design**
   - Connect:
     - Zynq PS M_AXI_GP / HP ports to AXI interconnects.
     - AXI DMA master/slave interfaces to:
       - PS / DDR (for data buffers).
       - Custom QCSNN IP (for input/output AXI-Stream).
     - Clock and reset nets from **Zynq PS** and **Processor System Reset** to AXI DMA, interconnects, and QCSNN IP.

3. **Generate bitstream and hardware handoff**
   - In Vivado:
     - **Validate** the block design (no connection or width errors).
     - Run **Synthesis** and then **Implementation**.
     - Generate:
       - **Bitstream** (`.bit`)
       - **Hardware handoff** (`.hwh` / `.xsa`, depending on flow).

4. **Deploy to PYNQ-Z2**
   - Copy the `.bit` and `.hwh` files to the board (e.g., `/home/xilinx/jupyter_notebooks/overlays/qcsnn/`).
   - From a PYNQ Python notebook:
     - Load the overlay.
     - Configure **AXI DMA** buffers for:
       - ECG segments in,
       - QCSNN predictions out.
     - Stream ECG data through the accelerator and collect:
       - Predictions,
       - Latency and throughput,
       - Optional power/energy measurements (external meter).

---

## 3. Vivado Block Design Components

The Vivado Block Design for this QCSNN accelerator uses the following major IP blocks:

- **Zynq-7000 Processing System (PS)**  
  - Zynq PS (Zynq7 Processing System) configures and controls the accelerator.  
  - Runs embedded Linux (PYNQ) and Python to:
    - Load the bitstream / overlay.
    - Configure AXI DMA.
    - Stream ECG segments to the QCSNN IP.
    - Collect predictions from DDR memory.

- **Custom QCSNN IP (HLS)**  
  - The synthesized HLS kernel implementing the **combined 2-stage QCSNN**:
    - Stage-1: binary Normal vs Abnormal.
    - Stage-2: 4-class arrhythmia classification for Abnormal beats.
  - Exposes AXI interfaces:
    - Typically **AXI4-Stream** for data in/out via DMA.
    - **AXI4-Lite** for configuration/control registers (if configured).

- **AXI DMA**
  - One or more **AXI Direct Memory Access (AXI DMA)** cores handle:
    - Streaming ECG input data from DDR to QCSNN IP (MM2S: Memory-Mapped to Stream).
    - Streaming prediction results back from QCSNN IP to DDR (S2MM: Stream to Memory-Mapped).
  - The PS configures DMA descriptors and initiates transfers.

- **AXI Interconnect / AXI Memory Interconnect**
  - **AXI Interconnect** (or AXI SmartConnect) connects:
    - Zynq PS master ports
    - AXI DMA
    - Custom QCSNN IP
  - **AXI Memory Interconnect** is used to connect:
    - High-performance (HP) ports of the Zynq PS to DDR,
    - AXI DMA master interfaces to DDR memory,
    - Ensuring sufficient bandwidth for streaming ECG data and collecting outputs.

- **Processor System Reset**
  - **Processor System Reset** IP ensures proper reset sequencing for:
    - AXI DMA,
    - QCSNN IP,
    - AXI interconnects,  
    based on PS reset signals and clocks.
  - This guarantees that all logic begins from a known, consistent state when the system boots or is reprogrammed.

- **`xlconstant` (constant IP)**
  - The **`xlconstant`** IP provides constant logic values (e.g., '0', '1') where needed:
    - Static enable signals,
    - Tie-offs for unused ports,
    - Constant configuration bits.
  - This avoids dangling/unconnected ports and simplifies design wiring.

- **Clocks & Resets**
  - Clocking is typically derived from the Zynq PS clock outputs.
  - Processor System Reset and `xlconstant` support clean reset/control of:
    - AXI DMA,
    - QCSNN IP,
    - AXI interconnects.

### Dataflow Summary

1. The **Zynq PS** (running Python/PYNQ) configures the **AXI DMA** and QCSNN IP.
2. ECG segments are stored in **DDR memory** attached to the PS.
3. AXI DMA (MM2S) streams ECG data from DDR → QCSNN IP via AXI4-Stream.
4. The **QCSNN custom IP** performs 2-stage spiking inference on the incoming data.
5. AXI DMA (S2MM) streams prediction results from QCSNN IP → DDR.
6. Zynq PS reads predictions from DDR and computes metrics / displays results.

---

## 4. Bitstream Generation and PYNQ Deployment

1. **Validate and generate bitstream in Vivado**
   - Validate the block design (Zynq PS + AXI DMA + QCSNN IP + interconnect + resets + `xlconstant`).
   - Generate the bitstream (`.bit`).
   - Export hardware handoff (`.hwh` / `.xsa` as needed for PYNQ).

2. **Transfer files to PYNQ-Z2**
   - Copy `.bit` and `.hwh` to the PYNQ filesystem, e.g.:
     - `/home/xilinx/jupyter_notebooks/overlays/qcsnn/`
   - Optionally include a small Python notebook/script to:
     - Load the overlay,
     - Configure DMA,
     - Stream ECG test segments,
     - Measure accuracy, latency, and power.

3. **Run inference on PYNQ-Z2**
   - From a PYNQ Jupyter notebook:
     - Import the overlay.
     - Allocate input/output buffers.
     - Push ECG segments through the accelerator.
     - Collect predictions and compute performance metrics.

---

## 5. High-Level Design Summary

- **Architecture**
  - Two-stage **Quantized Convolution Spiking Neural Network (QCSNN)**:
    - Stage-1 (binary): Normal vs Abnormal.
    - Stage-2 (multi-class): refined 4-class arrhythmia classification for Abnormal beats.
  - Quantization-aware training with **snnTorch** + **Brevitas**.

- **Quantization**
  - Fixed-point quantization of weights and activations (e.g., 8–12 bits).
  - C++ / HLS implementation mirrors the PyTorch quantization scheme.

- **Hardware**
  - Target platform: **PYNQ-Z2 (Zynq-7020)**.
  - Flow: PyTorch → C++ → Vitis HLS → Vivado (Zynq PS + AXI DMA + QCSNN IP + Interconnect) → PYNQ overlay.
  - Block design includes:
    - Zynq Processing System  
    - AXI DMA  
    - AXI Interconnect / AXI Memory Interconnect  
    - Custom QCSNN IP (HLS)  
    - Processor System Reset  
    - `xlconstant` for static signals  

---

## 6. Reproducibility Checklist

1. Prepare Python environment and install PyTorch, snnTorch, Brevitas, etc.  
2. Run Stage-1 and Stage-2 QCSNN notebooks and export quantized weights.  
3. Copy weights to `csnn_cpp/weights_sd/`.  
4. Build and test the C++ implementation on CPU.  
5. Use Vitis HLS to generate QCSNN IP.  
6. Build the Vivado block design (Zynq PS + AXI DMA + Interconnect + custom IP + reset + `xlconstant`).  
7. Generate bitstream + hardware handoff, deploy to PYNQ-Z2, and test.

---
## 7. Citation


```bibtex
@misc{qcsnn,
  author       = {Olamilekan Banjo and Behnaz Ghoraani},
  title        = {},
  year         = {2026},
  howpublished = {\url{https://github.com/BanjoOD/qcsnn}}
}
