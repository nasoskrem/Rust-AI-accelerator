# KRUMA — CPU AI Accelerator Core

![Version](https://img.shields.io/badge/version-0.1.2-blue)
![Rust](https://img.shields.io/badge/Made%20With-Rust-informational)
![Status](https://img.shields.io/badge/status-Prototype-yellow)

**KRUMA** is a low-level, high-performance **CPU-based AI accelerator** built in Rust, designed to mimic the abstractions and operations of GPU tensor libraries like CUDA or Metal — but without requiring dedicated hardware. It supports SIMD vectorization and multithreaded compute for fundamental AI operations like matrix multiplication and ReLU.

---

## ✨ Features (v0.1.2)

### ✅ Core Abstractions
- `DeviceBuffer` — High-performance memory buffer with both scalar and SIMD views.
- `HardwareBackend` — Modular backend trait for custom CPU/GPU execution engines.

### ✅ Implemented Backend
- **`CpuBackend`**:  
  - Multithreaded via `rayon`  
  - SIMD-accelerated via `wide::f32x4`  
  - Supports:
    - Matrix multiplication (`matmul`)
    - ReLU activation (`relu`, `relu_inplace`)

### ✅ SIMD & Parallelism
- Vectorized ops using `f32x4`
- Automatic parallelization with `rayon::par_iter`

---

## 📦 Usage Example

```rust
use kruma::{DeviceBuffer, CpuBackend, HardwareBackend};

let a = DeviceBuffer::from_slice(&[1.0, 2.0, 3.0, 4.0]);
let b = DeviceBuffer::from_slice(&[5.0, 6.0, 7.0, 8.0]);
let mut c = DeviceBuffer::new(4);

let backend = CpuBackend;
backend.matmul(&a, &b, &mut c);

println!("{:?}", c.as_slice());
