use crate::{Image, Label, TrainingImage};

/// A handle to a specific training [Label] from the MNIST dataset
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainingLabel(usize);
impl Label for TrainingLabel {
    /// Contents of `train-labels.idx1-ubyte"`
    const RAW_DATA: &[u8] = include_bytes!("../dataset/train-labels.idx1-ubyte");

    /// The number of training labels in MNIST. equal to [TestImage::COUNT]
    const COUNT: usize = TrainingImage::COUNT;

    unsafe fn from_index_unchecked(index: usize) -> Self {
        Self(index)
    }
    fn index(&self) -> usize {
        self.0
    }
}
