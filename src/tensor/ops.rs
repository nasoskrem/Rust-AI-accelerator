use crate::tensor::Tensor;

pub trait TensorOps<T> {
    fn add(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn matmul(&self, other: &Self) -> Self;
    fn relu(&self) -> Self;
}

impl TensorOps<f32> for Tensor<f32,2>{
    
    fn add(&self, other: &Self) -> Self {
        assert_eq!(self.shape, other.shape, "Shapes must match for addition");
        
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a,b)| a+b)
            .collect();
        Tensor {data, shape: self.shape, strides: self.strides }
    }

    fn mul(&self, other: &Self) -> Self {
        assert_eq!(self.shape, other.shape, "Shapes must match for mul");
        
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a,b)| a*b)
            .collect();
        Tensor {data, shape: self.shape, strides: self.strides }
    }

    fn relu(&self) -> Self {
        let data = self
            .data
            .iter()
            .map(|x| x.max(0.0))
            .collect();
        Tensor {data, shape: self.shape, strides: self.strides }
    }

    fn matmul(&self, other: &Self) -> Self {
        let [m,k1] = self.shape;
        let [k2,n] = other.shape;
        assert_eq!(k1,k2, "Inner dimensions must match");

        let mut out = Tensor::<f32,2>::new([m,n]);

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..k1 {
                    sum += self[[i, k]] * other[[k,j]];
                }
                out[[i,j]] = sum;
            }
        }
        out
    }
} 
