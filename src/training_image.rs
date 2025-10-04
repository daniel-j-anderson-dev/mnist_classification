use crate::Image;

/// A handle to a specific training [Image] from the MNIST dataset
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainingImage(usize);
impl Image for TrainingImage {
    /// Contents of `train-images.idx3-ubyte`
    const RAW_DATA: &[u8] = include_bytes!("../dataset/train-images.idx3-ubyte");

    /// The number of training images in MNIST
    const COUNT: usize = 60000;

    unsafe fn from_index_unchecked(index: usize) -> Self {
        Self(index)
    }
    fn index(&self) -> usize {
        self.0
    }
}
