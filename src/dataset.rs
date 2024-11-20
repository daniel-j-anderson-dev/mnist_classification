pub mod display_images;

pub const TRAINING_LABELS: &[u8] = include_bytes!("../../../MNIST/train-labels.idx1-ubyte");
pub const TRAINING_IMAGES: &[u8] = include_bytes!("../../../MNIST/train-images.idx3-ubyte");
pub const TEST_LABELS: &[u8] = include_bytes!("../../../MNIST/t10k-labels.idx1-ubyte");
pub const TEST_IMAGES: &[u8] = include_bytes!("../../../MNIST/t10k-images.idx3-ubyte");

pub const TRAINING_IMAGE_COUNT: usize = 60000;
pub const TEST_IMAGE_COUNT: usize = 10000;

pub const IMAGE_DATA_OFFSET: usize = 16;
pub const IMAGE_LABEL_OFFSET: usize = 8;

pub const IMAGE_WIDTH: usize = 28;
pub const IMAGE_HEIGHT: usize = 28;
pub const IMAGE_SIZE: usize = IMAGE_WIDTH * IMAGE_HEIGHT;

pub const LABELS_MAGIC_NUMBER: u32 = 2049;
pub const IMAGES_MAGIC_NUMBER: u32 = 2051;

const fn calculate_image_range(image_index: usize) -> std::ops::Range<usize> {
    let start = IMAGE_DATA_OFFSET + (image_index * IMAGE_SIZE);
    let end = start + IMAGE_SIZE;
    start..end
}

pub fn get_training_image(image_index: usize) -> &'static [u8; IMAGE_SIZE] {
    assert!(
        image_index < TRAINING_IMAGE_COUNT,
        "image index is out of bounds"
    );

    let image_range = calculate_image_range(image_index);
    (&TRAINING_IMAGES[image_range])
        .try_into()
        .expect("image_range.len() == IMAGE_SIZE")
}

pub fn get_test_image(image_index: usize) -> &'static [u8; IMAGE_SIZE] {
    assert!(
        image_index < TEST_IMAGE_COUNT,
        "image index is out of bounds"
    );

    let image_range = calculate_image_range(image_index);
    (&TEST_IMAGES[image_range])
        .try_into()
        .expect("image_range.len() == IMAGE_SIZE")
}

pub fn get_training_label(image_index: usize) -> u8 {
    assert!(image_index < TRAINING_IMAGE_COUNT);
    TRAINING_LABELS[image_index + IMAGE_LABEL_OFFSET]
}

pub fn get_test_label(image_index: usize) -> u8 {
    assert!(image_index < TEST_IMAGE_COUNT);
    TEST_LABELS[image_index + IMAGE_LABEL_OFFSET]
}

#[cfg(test)]
mod test {
    use crate::dataset::{
        get_test_image, get_test_label, get_training_image, get_training_label,
        IMAGES_MAGIC_NUMBER, IMAGE_HEIGHT, IMAGE_WIDTH, LABELS_MAGIC_NUMBER, TEST_IMAGES,
        TEST_IMAGE_COUNT, TEST_LABELS, TRAINING_IMAGES, TRAINING_IMAGE_COUNT, TRAINING_LABELS,
    };

    #[test]
    fn test_get_training_label() {
        assert_eq!(get_training_label(0), 5);
        assert_eq!(get_training_label(59999), 8);
    }
    
    #[test]
    fn test_get_test_label() {
        assert_eq!(get_test_label(0), 7);
        assert_eq!(get_test_label(9999), 6);
    }

    #[test]
    fn test_get_training_image() {
        for image_index in 0..TRAINING_IMAGE_COUNT {
            let _ = get_training_image(image_index);
        }
    }

    #[test]
    fn test_get_test_image() {
        for image_index in 0..TEST_IMAGE_COUNT {
            let _ = get_test_image(image_index);
        }
    }

    #[test]
    #[rustfmt::skip]
    fn dataset_sanity_check() {
        const fn u32_from_bytes(bytes: &[u8]) -> u32 {
            (bytes[0] as u32) << 24 | (bytes[1] as u32) << 16 | (bytes[2] as u32) << 8 | (bytes[3] as u32)
        }
    
        // https://yann.lecun.com/exdb/mnist/
        let training_labels_magic_number = u32_from_bytes(&TRAINING_LABELS[0..4]);
        let training_labels_count        = u32_from_bytes(&TRAINING_LABELS[4..8]);
        let training_images_magic_number = u32_from_bytes(&TRAINING_IMAGES[0..4]);
        let training_images_image_count  = u32_from_bytes(&TRAINING_IMAGES[4..8]);
        let training_images_height       = u32_from_bytes(&TRAINING_IMAGES[8..12]);
        let training_images_width        = u32_from_bytes(&TRAINING_IMAGES[12..16]);
        let test_labels_magic_number     = u32_from_bytes(&TEST_LABELS[0..4]);
        let test_labels_count            = u32_from_bytes(&TEST_LABELS[4..8]);
        let test_images_magic_number     = u32_from_bytes(&TEST_IMAGES[0..4]);
        let test_images_image_count      = u32_from_bytes(&TEST_IMAGES[4..8]);
        let test_images_height           = u32_from_bytes(&TEST_IMAGES[8..12]);
        let test_images_width            = u32_from_bytes(&TEST_IMAGES[12..16]);
        
        assert_eq!(training_labels_magic_number, LABELS_MAGIC_NUMBER);
        assert_eq!(training_labels_count,        TRAINING_IMAGE_COUNT as u32);
        assert_eq!(training_images_magic_number, IMAGES_MAGIC_NUMBER);
        assert_eq!(training_images_image_count,  TRAINING_IMAGE_COUNT as u32);
        assert_eq!(training_images_height,       IMAGE_HEIGHT as u32);
        assert_eq!(training_images_width,        IMAGE_WIDTH as u32);
        assert_eq!(test_labels_magic_number,     LABELS_MAGIC_NUMBER);
        assert_eq!(test_labels_count,            TEST_IMAGE_COUNT as u32);
        assert_eq!(test_images_magic_number,     IMAGES_MAGIC_NUMBER);
        assert_eq!(test_images_image_count,      TEST_IMAGE_COUNT as u32);
        assert_eq!(test_images_height,           IMAGE_HEIGHT as u32);
        assert_eq!(test_images_width,            IMAGE_WIDTH as u32);
    }
}
