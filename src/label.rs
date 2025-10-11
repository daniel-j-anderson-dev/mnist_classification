use crate::{Image, TestImage, TrainingImage};

/// The possible classes of digits in the MNIST dataset
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DigitClass {
    Zero,
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
}
impl DigitClass {
    /// The number of classes of digits
    pub const COUNT: usize = 10;
    pub const fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::Zero),
            1 => Some(Self::One),
            2 => Some(Self::Two),
            3 => Some(Self::Three),
            4 => Some(Self::Four),
            5 => Some(Self::Five),
            6 => Some(Self::Six),
            7 => Some(Self::Seven),
            8 => Some(Self::Eight),
            9 => Some(Self::Nine),
            _ => None,
        }
    }
    /// see https://en.wikipedia.org/wiki/One-hot
    #[rustfmt::skip]
    pub const fn one_hot_encode(self) -> [f32; 10] {
        match self {
            Self::Zero  => Self::ZERO,
            Self::One   => Self::ONE,
            Self::Two   => Self::TWO,
            Self::Three => Self::THREE,
            Self::Four  => Self::FOUR,
            Self::Five  => Self::FIVE,
            Self::Six   => Self::SIX,
            Self::Seven => Self::SEVEN,
            Self::Eight => Self::EIGHT,
            Self::Nine  => Self::NINE,
        }
    }

    pub const ZERO: [f32; DigitClass::COUNT] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    pub const ONE: [f32; DigitClass::COUNT] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    pub const TWO: [f32; DigitClass::COUNT] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    pub const THREE: [f32; DigitClass::COUNT] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    pub const FOUR: [f32; DigitClass::COUNT] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    pub const FIVE: [f32; DigitClass::COUNT] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    pub const SIX: [f32; DigitClass::COUNT] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    pub const SEVEN: [f32; DigitClass::COUNT] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    pub const EIGHT: [f32; DigitClass::COUNT] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    pub const NINE: [f32; DigitClass::COUNT] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
}
impl From<DigitClass> for usize {
    fn from(value: DigitClass) -> Self {
        value as usize
    }
}
impl From<[f32; DigitClass::COUNT]> for DigitClass {
    fn from(value: [f32; DigitClass::COUNT]) -> Self {
        match value {
            Self::ZERO => Self::Zero,
            Self::ONE => Self::One,
            Self::TWO => Self::Two,
            Self::THREE => Self::Three,
            Self::FOUR => Self::Four,
            Self::FIVE => Self::Five,
            Self::SIX => Self::Six,
            Self::SEVEN => Self::Seven,
            Self::EIGHT => Self::Eight,
            _ => Self::Nine,
        }
    }
}

/// The label data starts at byte `8` for the [TrainingLabel]s and [TestLabel]s
pub const LABEL_OFFSET: usize = 8;
/// Each labels file starts with `2049`
pub const LABEL_MAGIC_NUMBER: u32 = 2049;

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

    fn all() -> impl Iterator<Item = Self> {
        (0..Self::COUNT).filter_map(Self::from_index)
    }
}

/// A handle to a specific test [Label] from the MNIST dataset
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TestLabel(usize);
impl Label for TestLabel {
    /// Contents of `t10k-labels.idx1-ubyte`
    const RAW_DATA: &[u8] = include_bytes!("../dataset/t10k-labels.idx1-ubyte");

    /// The number of test labels in MNIST. equal to [TestImage::COUNT]
    const COUNT: usize = TestImage::COUNT;

    unsafe fn from_index_unchecked(index: usize) -> Self {
        Self(index)
    }
    fn index(&self) -> usize {
        self.0
    }
}

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
