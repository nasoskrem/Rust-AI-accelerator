use wide::f32x4;

pub mod cpu;

pub struct DeviceBuffer {
    data: Vec<f32>,
}

impl DeviceBuffer {
    pub fn new(size: usize) -> Self {
        let mut data: Vec<f32> = Vec::with_capacity(size);
        unsafe { data.set_len(size) };
        Self { data }
    }

    pub fn from_slice(slice: &[f32]) -> Self {
        Self {
            data: slice.to_vec(),
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn as_simd(&self) -> &[f32x4] {
        assert!(self.data.len() % 4 == 0, "Data length must be multiple of 4");
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const f32x4,
                self.data.len() / 4,
            )
        }
    }

    pub fn as_mut_simd(&mut self) -> &mut [f32x4] {
        assert!(self.data.len() % 4 == 0, "Data length must be multiple of 4");
        unsafe {
            std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr() as *mut f32x4,
                self.data.len() / 4,
            )
        }
    }
}

pub trait HardwareBackend {
    fn matmul(&self, a: &DeviceBuffer, b: &DeviceBuffer, c: &mut DeviceBuffer);
    fn relu(&self, input: &DeviceBuffer, output: &mut DeviceBuffer);
    fn relu_inplace(&self, data: &mut DeviceBuffer);
}