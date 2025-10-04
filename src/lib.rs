//! https://yann.lecun.com/exdb/mnist/

pub mod digit_class;
pub mod image;
pub mod label;
pub mod test_image;
pub mod test_label;
pub mod training_image;
pub mod training_label;
pub mod visualization;

pub use crate::{
    digit_class::DigitClass, image::Image, label::Label, test_image::TestImage,
    test_label::TestLabel, training_image::TrainingImage, training_label::TrainingLabel,
};

#[cfg(test)]
mod test {
    use crate::{
        image::{IMAGE_HEIGHT, IMAGE_MAGIC_NUMBER, IMAGE_WIDTH},
        label::LABEL_MAGIC_NUMBER,
        Image, Label, TestImage, TestLabel, TrainingImage, TrainingLabel,
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
            IMAGE_MAGIC_NUMBER
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
            LABEL_MAGIC_NUMBER
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
            IMAGE_MAGIC_NUMBER
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
            LABEL_MAGIC_NUMBER
        );
        assert_eq!(
            u32_from_big_endian_bytes(&TrainingLabel::RAW_DATA[4..8]),
            TrainingImage::COUNT as u32
        );
    }
}
