use crate::tensor::Tensor;
use std::sync::Arc;

#[derive(Debug)]
pub enum KernelOp {
    MatMul {
        a: Arc<Tensor<f32, 2>>,
        b: Arc<Tensor<f32, 2>>,
        out: Arc<Tensor<f32, 2>>,
    },
    ReLU {
        input: Arc<Tensor<f32, 2>>,
        out: Arc<Tensor<f32, 2>>,
    },
}

pub trait KernelExecutor {
    fn execute(&self, op: &KernelOp);
}