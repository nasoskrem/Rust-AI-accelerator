use kruma::backend::CpuBackend;
use kruma::tensor::DeviceBuffer;
use kruma::backend::HardwareBackend;
use kruma::tensor::Tensor;
use kruma::tensor::ops::TensorOps;


fn main() {
    {
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
    {
        let a = Tensor::from_data([2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = Tensor::from_data([2, 2], vec![5.0, 6.0, 7.0, 8.0]);

        // Addition
        let sum = a.add(&b);
        println!("Addition:\n{:?}", sum);

        // Multiplication
        let mul = a.mul(&b);
        println!("Multiplication:\n{:?}", mul);

        // Matrix Multiplication
        let matmul = a.matmul(&b);
        println!("Matrix Multiplication:\n{:?}", matmul);

        // ReLU
        let c = Tensor::from_data([2, 2], vec![-1.0, 2.0, -3.0, 4.0]);
        let relu = c.relu();
        println!("ReLU:\n{:?}", relu);
    }

}