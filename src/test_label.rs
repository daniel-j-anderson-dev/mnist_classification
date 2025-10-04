use crate::{Image, Label, TestImage};

/// A handle to a specific test [Label] from the MNIST dataset
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TestLabel(usize);
impl Label for TestLabel {
    /// Contents of `t10k-labels.idx1-ubyte`
    const RAW_DATA: &[u8] = include_bytes!("../dataset/t10k-labels.idx1-ubyte");

    /// The number of test labels in MNIST. equal to [TestImage::COUNT]
    const COUNT: usize = TestImage::COUNT;
    
    unsafe fn from_index_unchecked(index: usize) -> Self {
        Self(index)
    }
    fn index(&self) -> usize {
        self.0
    }
}
