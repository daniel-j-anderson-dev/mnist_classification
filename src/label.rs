use crate::DigitClass;

/// The label data starts at byte `8` for the [TrainingLabel]s and [TestLabel]s
pub const LABEL_OFFSET: usize = 8;
/// Each labels file starts with `2049`
pub const LABELS_MAGIC_NUMBER: u32 = 2049;

/// A handle to a specific label from the MNIST dataset
pub trait Label: Sized {
    const RAW_DATA: &[u8];
    const COUNT: usize;
    /// # Safety:
    /// `index < <Self as Image>::COUNT` must be `true`
    unsafe fn from_index_unchecked(index: usize) -> Self;
    fn index(&self) -> usize;

    fn from_index(index: usize) -> Option<Self> {
        (index < Self::COUNT).then(|| unsafe { Self::from_index_unchecked(index) })
    }

    fn digit_class(&self) -> DigitClass {
        let index = self.index() + LABEL_OFFSET;
        DigitClass::from_byte(Self::RAW_DATA[index])
            .expect("all bytes after LABEL_OFFSET are in range 0..=9")
    }

    fn one_hot_encode_from_index(index: usize) -> Option<[f64; DigitClass::COUNT]> {
        Self::from_index(index).map(|label| label.digit_class().one_hot_encode())
    }

    fn all() -> impl Iterator<Item = Self> {
        (0..Self::COUNT).filter_map(Self::from_index)
    }

    fn all_one_hot_encoded() -> impl Iterator<Item = [f64; DigitClass::COUNT]> {
        (0..Self::COUNT).filter_map(Self::one_hot_encode_from_index)
    }
}
