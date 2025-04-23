use crate::tensor::DeviceBuffer;

pub trait HardwareBackend {
    fn matmul(&self, a: &DeviceBuffer, b: &DeviceBuffer, c: &mut DeviceBuffer);
    fn relu(&self, input: &DeviceBuffer, output: &mut DeviceBuffer);
    fn relu_inplace(&self, data: &mut DeviceBuffer);
}