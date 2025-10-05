//! # FILE FORMATS FOR THE MNIST DATABASE
//! The data is stored in a very simple file format designed for storing vectors and multidimensional matrices. General info on this format is given
//! at the end of this page, but you don't need to read that to use the data files.
//!
//! All the integers in the files are stored in the MSB first (high endian) format used by most non-Intel processors. Users of Intel processors and
//! other low-endian machines must flip the bytes of the header.
//!
//! There are 4 files:
//!
//! - `train-images-idx3-ubyte`: training set images
//! - `train-labels-idx1-ubyte`: training set labels
//! - `t10k-images-idx3-ubyte`:  test set images
//! - `t10k-labels-idx1-ubyte`:  test set labels
//!
//! The training set contains `60000` examples, and the test set `10000` examples.
//!
//! The first `5000` examples of the test set are taken from the original NIST training set. The last `5000` are taken from the original NIST test
//! set. The first `5000` are cleaner and easier than the last `5000`.
//!
//! ## TRAINING SET LABEL FILE (`train-labels-idx1-ubyte`):
//! | offset | type           | value            | description              |
//! |--------|----------------|------------------|--------------------------|
//! | 0000   | 32 bit integer | 0x00000801(2049) | magic number (MSB first) |
//! | 0004   | 32 bit integer | 60000            | number of items          |
//! | 0008   | unsigned byte  | ??               | label                    |
//! | 0009   | unsigned byte  | ??               | label                    |
//! | ...    | ...            | ...              | ...                      |
//! | xxxx   | unsigned byte  | ??               | label                    |
//!
//! The labels values are `0` to `9`.
//!
//! ## TRAINING SET IMAGE FILE (`train-images-idx3-ubyte`):
//! | offset | type           | value            | description       |
//! |--------|----------------|------------------|-------------------|
//! | 0000   | 32 bit integer | 0x00000803(2051) | magic number      |
//! | 0004   | 32 bit integer | 60000            | number of images  |
//! | 0008   | 32 bit integer | 28               | number of rows    |
//! | 0012   | 32 bit integer | 28               | number of columns |
//! | 0016   | unsigned byte  | ??               | pixel             |
//! | 0017   | unsigned byte  | ??               | pixel             |
//! | ...    | ...            | ...              | ...               |
//! | xxxx   | unsigned byte  | ??               | pixel             |
//!                 
//! Pixels are organized row-wise. Pixel values are `0` to `255`. `0` means background (white), `255` means foreground (black).
//!
//! ## TEST SET LABEL FILE (`t10k-labels-idx1-ubyte`):
//! | offset | type           | value            | description              |
//! |--------|----------------|------------------|--------------------------|
//! | 0000   | 32 bit integer | 0x00000801(2049) | magic number (MSB first) |
//! | 0004   | 32 bit integer | 10000            | number of items          |
//! | 0008   | unsigned byte  | ??               | label                    |
//! | 0009   | unsigned byte  | ??               | label                    |
//! | ...    | ...            | ...              | ...                      |
//! | xxxx   | unsigned byte  | ??               | label                    |
//!
//! The labels values are `0` to `9`.
//!
//! ## TEST SET IMAGE FILE (`t10k-images-idx3-ubyte`):
//! | offset | type           | value            | description       |
//! |--------|----------------|------------------|-------------------|
//! | 0000   | 32 bit integer | 0x00000803(2051) | magic number      |
//! | 0004   | 32 bit integer | 10000            | number of images  |
//! | 0008   | 32 bit integer | 28               | number of rows    |
//! | 0012   | 32 bit integer | 28               | number of columns |
//! | 0016   | unsigned byte  | ??               | pixel             |
//! | 0017   | unsigned byte  | ??               | pixel             |
//! | ...    | ...            | ...              | ...               |
//! | xxxx   | unsigned byte  | ??               | pixel             |
//!
//! Pixels are organized row-wise. Pixel values are `0` to `255`. `0` means background (white), `255` means foreground (black).
//!
//! # THE IDX FILE FORMAT
//! the IDX file format is a simple format for vectors and multidimensional matrices of various numerical types.
//!
//! The basic format is:
//! ```text
//! magic number
//! size in dimension 0
//! size in dimension 1
//! size in dimension 2
//! .....
//! size in dimension N
//! data
//! ```
//!
//! - The magic number is an integer (**MSB first**).
//!   - The **first** 2 bytes are always 0.
//! - The **third byte** codes the type of the data:
//!   - `0x08`: unsigned byte
//!   - `0x09`: signed byte
//!   - `0x0B`: short (2 bytes)
//!   - `0x0C`: int (4 bytes)
//!   - `0x0D`: float (4 bytes)
//!   - `0x0E`: double (8 bytes)
//! - The **fourth byte** codes the number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices....
//! - The sizes in each dimension are **4-byte** integers (**MSB first**, high endian, like in most non-Intel processors).
//! - The data is stored like in a C array, i.e. the index in the last dimension changes the fastest.

pub mod image;
pub mod label;
pub mod visualization;

pub use crate::{image::*, label::*};

pub struct Datum {
    pub input: [f32; IMAGE_SIZE],
    pub expected_output: [f32; DigitClass::COUNT],
    pub digit_class: DigitClass,
}

pub trait DataSet {
    type Image: Image;
    type Label: Label;
    const COUNT: usize = Self::Image::COUNT;
    fn images() -> impl Iterator<Item = [f32; IMAGE_SIZE]> {
        Self::Image::all().map(|image| normalize_bytes(image.as_bytes()))
    }
    fn labels() -> impl Iterator<Item = [f32; DigitClass::COUNT]> {
        Self::Label::all().map(|label| label.digit_class().one_hot_encode())
    }
    fn all() -> impl Iterator<Item = Datum> {
        Self::Image::all()
            .zip(Self::Label::all())
            .map(|(image, label)| Datum {
                input: normalize_bytes(image.as_bytes()),
                expected_output: label.digit_class().one_hot_encode(),
                digit_class: label.digit_class(),
            })
    }
}
pub enum TrainingData {}
impl DataSet for TrainingData {
    type Image = TrainingImage;
    type Label = TrainingLabel;
}
pub enum TestData {}
impl DataSet for TestData {
    type Image = TestImage;
    type Label = TestLabel;
}

#[cfg(test)]
mod test {
    use crate::{
        DataSet, Image, Label, TestData, TestImage, TestLabel, TrainingData, TrainingImage,
        TrainingLabel,
        image::{IMAGE_HEIGHT, IMAGE_MAGIC_NUMBER, IMAGE_WIDTH},
        label::LABEL_MAGIC_NUMBER,
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

    #[test]
    fn test_counts() {
        assert_eq!(TestData::images().count(), TestImage::COUNT);
        assert_eq!(TestData::labels().count(), TestLabel::COUNT);
        assert_eq!(TrainingData::images().count(), TrainingImage::COUNT);
        assert_eq!(TrainingData::labels().count(), TrainingLabel::COUNT);
    }
}
