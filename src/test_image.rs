use crate::Image;

/// A handle to a specific test [Image] from the MNIST dataset
pub struct TestImage(usize);
impl Image for TestImage {
    const RAW_DATA: &[u8] = include_bytes!("../dataset/t10k-images.idx3-ubyte");
    const COUNT: usize = 10000;
    unsafe fn from_index_unchecked(index: usize) -> Self {
        Self(index)
    }
    fn index(&self) -> usize {
        self.0
    }
}
