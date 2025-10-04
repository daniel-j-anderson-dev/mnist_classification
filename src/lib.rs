//! https://yann.lecun.com/exdb/mnist/

pub mod digit_class;
pub mod display;
pub mod test_image;
pub mod test_label;
pub mod training_image;
pub mod training_label;

pub use crate::{
    digit_class::DigitClass, test_image::TestImage, test_label::TestLabel,
    training_image::TrainingImage, training_label::TrainingLabel,
};

/// The image data starts at byte `16` for the [TrainingLabel]s and [TestLabel]s
pub const IMAGE_OFFSET: usize = 16;
/// The label data starts at byte `8` for the [TrainingLabel]s and [TestLabel]s
pub const LABEL_OFFSET: usize = 8;

/// Each image is `28 pixels` wide
pub const IMAGE_WIDTH: usize = 28;
/// Each image is `28 pixels` heigh
pub const IMAGE_HEIGHT: usize = 28;
/// Each image has a resolution of `28x28 pixels^2`
pub const IMAGE_SIZE: usize = IMAGE_WIDTH * IMAGE_HEIGHT;

/// Each labels file starts with `2049`
pub const LABELS_MAGIC_NUMBER: u32 = 2049;
/// Each images file starts with `2051`
pub const IMAGES_MAGIC_NUMBER: u32 = 2051;

pub const fn calculate_image_bounds(image_index: usize) -> (usize, usize) {
    let start = IMAGE_OFFSET + (image_index * IMAGE_SIZE);
    let end = start + IMAGE_SIZE;
    (start, end)
}

/// A handle to a specific image from the MNIST dataset
pub trait Image: Sized {
    const RAW_DATA: &[u8];
    const COUNT: usize;
    /// # Safety:
    /// `index < <Self as Image>::COUNT` must be `true`
    unsafe fn from_index_unchecked(index: usize) -> Self;
    fn index(&self) -> usize;

    fn from_index(index: usize) -> Option<Self> {
        (index < Self::COUNT).then(|| unsafe { Self::from_index_unchecked(index) })
    }
    fn as_bytes(&self) -> &'static [u8; IMAGE_SIZE] {
        let (start, end) = calculate_image_bounds(self.index());
        Self::RAW_DATA[start..end]
            .try_into()
            .expect("end - start == IMAGE_SIZE")
    }
    fn normalized(&self) -> [f64; IMAGE_SIZE] {
        let image_data = self.as_bytes();
        core::array::from_fn::<_, IMAGE_SIZE, _>(|i| image_data[i] as f64 / 255.0)
    }
    fn normalized_from_index(index: usize) -> Option<[f64; IMAGE_SIZE]> {
        Self::from_index(index).map(|s| s.normalized())
    }
    fn all() -> impl Iterator<Item = Self> {
        (0..Self::COUNT).filter_map(Self::from_index)
    }
    fn all_normalized() -> impl Iterator<Item = [f64; IMAGE_SIZE]> {
        (0..Self::COUNT).filter_map(Self::normalized_from_index)
    }
}

/// A handle to a specific label from the MNIST dataset
pub trait Label: Sized {
    const COUNT: usize;
    const RAW_DATA: &[u8];
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
    /// returns an iterator that yields all the labels from the MNIST dataset after being [DigitClass::one_hot_encode]d. the order is the same as Image::all_normalized
    fn all_one_hot_encoded() -> impl Iterator<Item = [f64; DigitClass::COUNT]> {
        (0..Self::COUNT).filter_map(Self::one_hot_encode_from_index)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        Image, Label, TestImage, TestLabel, TrainingImage, TrainingLabel, IMAGES_MAGIC_NUMBER,
        IMAGE_HEIGHT, IMAGE_WIDTH, LABELS_MAGIC_NUMBER,
    };

    const fn u32_from_big_endian_bytes(bytes: &[u8]) -> u32 {
        (bytes[0] as u32) << 24
            | (bytes[1] as u32) << 16
            | (bytes[2] as u32) << 8
            | (bytes[3] as u32)
    }

    #[test]
    fn test_image_metadata() {
        assert_eq!(
            u32_from_big_endian_bytes(&TestImage::RAW_DATA[0..4]),
            IMAGES_MAGIC_NUMBER
        );
        assert_eq!(
            u32_from_big_endian_bytes(&TestImage::RAW_DATA[4..8]),
            TestImage::COUNT as u32
        );
        assert_eq!(
            u32_from_big_endian_bytes(&TestImage::RAW_DATA[8..12]),
            IMAGE_HEIGHT as u32
        );
        assert_eq!(
            u32_from_big_endian_bytes(&TestImage::RAW_DATA[12..16]),
            IMAGE_WIDTH as u32
        );
    }

    #[test]
    fn test_label_metadata() {
        assert_eq!(
            u32_from_big_endian_bytes(&TestLabel::RAW_DATA[0..4]),
            LABELS_MAGIC_NUMBER
        );
        assert_eq!(
            u32_from_big_endian_bytes(&TestLabel::RAW_DATA[4..8]),
            TestImage::COUNT as u32
        );
    }

    #[test]
    fn training_image_metadata() {
        assert_eq!(
            u32_from_big_endian_bytes(&TrainingImage::RAW_DATA[0..4]),
            IMAGES_MAGIC_NUMBER
        );
        assert_eq!(
            u32_from_big_endian_bytes(&TrainingImage::RAW_DATA[4..8]),
            TrainingImage::COUNT as u32
        );
        assert_eq!(
            u32_from_big_endian_bytes(&TrainingImage::RAW_DATA[8..12]),
            IMAGE_HEIGHT as u32
        );
        assert_eq!(
            u32_from_big_endian_bytes(&TrainingImage::RAW_DATA[12..16]),
            IMAGE_WIDTH as u32
        );
    }

    #[test]
    fn training_label_metadata() {
        assert_eq!(
            u32_from_big_endian_bytes(&TrainingLabel::RAW_DATA[0..4]),
            LABELS_MAGIC_NUMBER
        );
        assert_eq!(
            u32_from_big_endian_bytes(&TrainingLabel::RAW_DATA[4..8]),
            TrainingImage::COUNT as u32
        );
    }
}
