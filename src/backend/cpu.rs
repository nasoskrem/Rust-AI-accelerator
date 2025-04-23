use crate::tensor::DeviceBuffer;
use crate::backend::HardwareBackend;
use rayon::prelude::*;
use wide::f32x4;

pub struct CpuBackend;

impl HardwareBackend for CpuBackend {
    fn matmul(&self, a: &DeviceBuffer, b: &DeviceBuffer, c: &mut DeviceBuffer) {
        let n = (a.as_slice().len() as f32).sqrt() as usize;
        let a_slice = a.as_slice();
        let b_slice = b.as_slice();
        let c_slice = c.as_mut_slice();
    
        assert_eq!(a_slice.len(), b_slice.len());
        assert_eq!(a_slice.len(), c_slice.len());
        assert!(n * n == a_slice.len());
    
        c_slice.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
            for j in 0..n {
                let mut sum = 0.0;
                let mut k = 0;
    
                while k + 4 <= n {
                    let a_chunk = f32x4::new([
                        a_slice[i * n + k],
                        a_slice[i * n + k + 1],
                        a_slice[i * n + k + 2],
                        a_slice[i * n + k + 3],
                    ]);
                    
                    let b_chunk = f32x4::new([
                        b_slice[k * n + j],
                        b_slice[(k + 1) * n + j],
                        b_slice[(k + 2) * n + j],
                        b_slice[(k + 3) * n + j],
                    ]);
    
                    sum += (a_chunk * b_chunk).reduce_add();
                    k += 4;
                }
    
                while k < n {
                    sum += a_slice[i * n + k] * b_slice[k * n + j];
                    k += 1;
                }
    
                c_row[j] = sum;
            }
        });
    }

    fn relu_inplace(&self, data: &mut DeviceBuffer) {
        data.as_mut_slice().iter_mut().for_each(|x| *x = x.max(0.0));
    }

    fn relu(&self, input: &DeviceBuffer, output: &mut DeviceBuffer) {
        input.as_slice()
            .par_iter()
            .zip(output.as_mut_slice().par_iter_mut())
            .for_each(|(&x, o)| {
                *o = x.max(0.0);
            });
    }
}
