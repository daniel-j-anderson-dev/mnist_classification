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
    pub const fn one_hot_encode(self) -> [f32; 10] {
        let mut encoded = [0.0; 10];
        encoded[self as usize] = 1.0;
        encoded
    }
}
impl From<DigitClass> for usize {
    fn from(value: DigitClass) -> Self {
        value as usize
    }
}
