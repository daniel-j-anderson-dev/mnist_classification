use crate::Image;

/// A handle to a specific training [Image] from the MNIST dataset
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainingImage(usize);
impl Image for TrainingImage {
    const RAW_DATA: &[u8] = include_bytes!("../../dataset/train-images.idx3-ubyte");
    const COUNT: usize = 60000;
    unsafe fn from_index_unchecked(index: usize) -> Self {
        Self(index)
    }
    fn index(&self) -> usize {
        self.0
    }
}
