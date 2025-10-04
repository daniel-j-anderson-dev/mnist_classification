use crate::{Image, Label, TestImage};

/// A handle to a specific test [Label] from the MNIST dataset
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TestLabel(usize);
impl Label for TestLabel {
    const COUNT: usize = TestImage::COUNT;
    const RAW_DATA: &[u8] = include_bytes!("../../dataset/t10k-labels.idx1-ubyte");
    unsafe fn from_index_unchecked(index: usize) -> Self {
        Self(index)
    }
    fn index(&self) -> usize {
        self.0
    }
}
