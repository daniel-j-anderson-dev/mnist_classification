/// The image data starts at byte `16` for the [TrainingLabel]s and [TestLabel]s
pub const IMAGE_OFFSET: usize = 16;

/// Each image is `28 pixels` wide
pub const IMAGE_WIDTH: usize = 28;

/// Each image is `28 pixels` heigh
pub const IMAGE_HEIGHT: usize = 28;

/// Each image has a resolution of `28x28 pixels^2`
pub const IMAGE_SIZE: usize = IMAGE_WIDTH * IMAGE_HEIGHT;

/// Each images file starts with `2051`
pub const IMAGE_MAGIC_NUMBER: u32 = 2051;

/// Returns `(start, end)`
pub const fn calculate_image_bounds(image_index: usize) -> (usize, usize) {
    let start = IMAGE_OFFSET + (image_index * IMAGE_SIZE);
    let end = start + IMAGE_SIZE;
    (start, end)
}

fn normalize_byte(b: u8) -> f32 {
    b as f32 / u8::MAX as f32
}

pub(super) fn normalize_bytes<const N: usize>(bytes: &[u8; N]) -> [f32; N] {
    core::array::from_fn(|i| normalize_byte(bytes[i]))
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

    fn as_bytes(self) -> &'static [u8; IMAGE_SIZE] {
        let (start, end) = calculate_image_bounds(self.index());
        Self::RAW_DATA[start..end]
            .try_into()
            .expect("end - start == IMAGE_SIZE")
    }

    fn all() -> impl Iterator<Item = Self> {
        (0..Self::COUNT).filter_map(Self::from_index)
    }
}

/// A handle to a specific test [Image] from the MNIST dataset
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TestImage(usize);
impl Image for TestImage {
    /// Contents of `t10k-images.idx3-ubyte`
    const RAW_DATA: &[u8] = include_bytes!("../dataset/t10k-images.idx3-ubyte");

    /// The number of test images in MNIST
    const COUNT: usize = 10000;

    unsafe fn from_index_unchecked(index: usize) -> Self {
        Self(index)
    }
    fn index(&self) -> usize {
        self.0
    }
}

/// A handle to a specific training [Image] from the MNIST dataset
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainingImage(usize);
impl Image for TrainingImage {
    /// Contents of `train-images.idx3-ubyte`
    const RAW_DATA: &[u8] = include_bytes!("../dataset/train-images.idx3-ubyte");

    /// The number of training images in MNIST
    const COUNT: usize = 60000;

    unsafe fn from_index_unchecked(index: usize) -> Self {
        Self(index)
    }
    fn index(&self) -> usize {
        self.0
    }
}
