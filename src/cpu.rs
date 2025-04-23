use super::{DeviceBuffer, HardwareBackend};
use rayon::prelude::*;
use wide::f32x4;

pub struct CpuBackend;

impl HardwareBackend for CpuBackend {
    fn matmul(&self, a: &DeviceBuffer, b: &DeviceBuffer, c: &mut DeviceBuffer) {
        let n = (a.as_slice().len() as f32).sqrt() as usize;
        
        if n < 2 {
            for i in 0..n {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in 0..n {
                        sum += a.as_slice()[i * n + k] * b.as_slice()[k * n + j];
                    }
                    c.as_mut_slice()[i * n + j] = sum;
                }
            }
            return;
        }

        // For larger matrices, use SIMD
        let a_simd = a.as_simd();
        let b_simd = b.as_simd();
        let c_simd = c.as_mut_simd();
        

        c_simd.par_iter_mut()
            .enumerate()
            .for_each(|(idx, c_row)| {
                let i = idx / n;
                let j = idx % n;
                let mut sum = f32x4::ZERO;
                for k in 0..n {
                    sum += a_simd[i * n + k] * b_simd[k * n + j];
                }
                *c_row = sum;
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