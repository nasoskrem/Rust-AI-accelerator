extern crate kruma;
use kruma::tensor::Tensor;
use kruma::tensor::ops::TensorOps;


//very heavy experiment!!
fn main(){
    let size = 1000;
    let mut data = vec![0.0; size * size];
    for i in 0..size {
        for j in 0..size {
            data[i * size + j] = (i + j) as f32 / size as f32;
        }
    }

    let a = Tensor::from_data([size, size], data.clone());
    let b = Tensor::from_data([size, size], data);

    let start = std::time::Instant::now();
    let matmul = a.matmul(&b);
    let duration = start.elapsed();
    println!("MatMul ({}x{}) took {:?}", size, size, duration);

    // Verify a few elements
    println!("Sample result [0,0]: {}", matmul[[0, 0]]);
    println!("Sample result [500,500]: {}", matmul[[500, 500]]);
}