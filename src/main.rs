use kruma::backend::CpuBackend;
use kruma::tensor::DeviceBuffer;
use kruma::backend::HardwareBackend;


fn main() {
    let backend = CpuBackend;

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