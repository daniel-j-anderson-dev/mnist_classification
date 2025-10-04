use crate::Image;

/// A handle to a specific test [Image] from the MNIST dataset
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TestImage(usize);
impl Image for TestImage {
    /// Contents of `t10k-images.idx3-ubyte`
    const RAW_DATA: &[u8] = include_bytes!("../dataset/t10k-images.idx3-ubyte");

    /// The number of test images in MNIST
    const COUNT: usize = 10000;

    unsafe fn from_index_unchecked(index: usize) -> Self {
        Self(index)
    }
    fn index(&self) -> usize {
        self.0
    }
}
