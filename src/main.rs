// src/main.rs
use kruma::{DeviceBuffer, HardwareBackend, cpu::CpuBackend};

fn main() {
    let backend = CpuBackend;

    // Test with 4x4 matrices (16 elements) to work with f32x4
    let a = DeviceBuffer::from_slice(&[
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    ]);
    
    let b = DeviceBuffer::from_slice(&[
        4.0, 3.0, 2.0, 1.0,
        4.0, 3.0, 2.0, 1.0,
        4.0, 3.0, 2.0, 1.0,
        4.0, 3.0, 2.0, 1.0
    ]);
    
    let mut c = DeviceBuffer::new(16); 

    println!("Running matrix multiplication...");
    backend.matmul(&a, &b, &mut c);
    
    println!("Matrix multiplication result:");
    for i in 0..4 {
        for j in 0..4 {
            print!("{:5.2} ", c.as_slice()[i * 4 + j]);
        }
        println!();
    }

    println!("\nApplying ReLU...");
    backend.relu_inplace(&mut c);
    
    println!("After ReLU:");
    for i in 0..4 {
        for j in 0..4 {
            print!("{:5.2} ", c.as_slice()[i * 4 + j]);
        }
        println!();
    }
}