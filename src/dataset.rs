pub mod display_images;

pub const TRAINING_LABELS: &[u8] = include_bytes!("../../../MNIST/train-labels.idx1-ubyte");
pub const TRAINING_IMAGES: &[u8] = include_bytes!("../../../MNIST/train-images.idx3-ubyte");
pub const TEST_LABELS: &[u8] = include_bytes!("../../../MNIST/t10k-labels.idx1-ubyte");
pub const TEST_IMAGES: &[u8] = include_bytes!("../../../MNIST/t10k-images.idx3-ubyte");

pub const TRAINING_IMAGE_COUNT: usize = 60000;
pub const TEST_IMAGE_COUNT: usize = 10000;

pub const IMAGE_DATA_OFFSET: usize = 16;
pub const IMAGE_WIDTH: usize = 28;
pub const IMAGE_HEIGHT: usize = 28;

pub const LABELS_MAGIC_NUMBER: u32 = 2049;
pub const IMAGES_MAGIC_NUMBER: u32 = 2051;

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
