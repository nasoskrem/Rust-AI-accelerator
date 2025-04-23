pub mod kernel;  
use crate::graph::kernel::{KernelExecutor, KernelOp};

#[derive(Default)]
pub struct Graph {
    pub ops: Vec<KernelOp>,
}

impl Graph {
    pub fn new() -> Self {
        Self { ops: vec![] }
    }

    pub fn add_op(&mut self, op: KernelOp) {
        self.ops.push(op);
    }

    pub fn run<E: KernelExecutor>(&self, executor: &E) {
        for op in &self.ops {
            executor.execute(op);
        }
    }
}