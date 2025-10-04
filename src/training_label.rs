use crate::{Image, Label, TrainingImage};

/// A handle to a specific training [Label] from the MNIST dataset
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainingLabel(usize);
impl Label for TrainingLabel {
    const COUNT: usize = TrainingImage::COUNT;
    const RAW_DATA: &[u8] = include_bytes!("../dataset/train-labels.idx1-ubyte");
    unsafe fn from_index_unchecked(index: usize) -> Self {
        Self(index)
    }
    fn index(&self) -> usize {
        self.0
    }
}
