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

    /// Returns the image data where each pixel scaled to be on the range `0.0..=1.0`
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
