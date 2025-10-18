use crate::*;

use core::marker::PhantomData;

use burn::{data::dataloader::batcher::Batcher, prelude::*};

/// - The `0th` index of each [Tensor]: indexes which image/label in this batch
/// - The `1th` index of each `images`: indexes which row of pixels in this image
/// - The `2th` index of each `images`: indexes which pixel in this row
#[derive(Debug, Clone)]
pub struct MnistBatch<B: Backend> {
    /// Shape: `[batch_size, image_height, image_width]`
    pub images: Tensor<B, 3>,

    /// Shape: `[batch_size]`
    pub labels: Tensor<B, 1, Int>,
}

#[derive(Debug, Clone)]
pub struct MnistBatcher;
impl<B, Image, Label> Batcher<B, (Image, Label), MnistBatch<B>> for MnistBatcher
where
    B: Backend,
    Image: crate::Image,
    Label: crate::Label,
{
    fn batch(&self, items: Vec<(Image, Label)>, device: &<B as Backend>::Device) -> MnistBatch<B> {
        let images = items
            .iter()
            .map(|(image, _)| image.to_array())
            .map(|data| Tensor::<B, 2>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, 28, 28]))
            .collect();

        let labels = items
            .iter()
            .map(|(_, label)| [(label.digit_class() as i64).elem::<B::IntElem>()])
            .map(|data| Tensor::<B, 1, Int>::from_data(data, device))
            .collect();

        MnistBatch {
            images: Tensor::cat(images, 0),
            labels: Tensor::cat(labels, 0),
        }
    }
}

pub struct MnistDataset<D: DataSet> {
    items: Vec<(D::Image, D::Label)>,
    _marker: PhantomData<D>,
}
impl<D: DataSet> MnistDataset<D> {
    pub fn new() -> Self {
        Self {
            items: D::all().collect(),
            _marker: PhantomData,
        }
    }
}
impl<D> burn::data::dataloader::Dataset<(D::Image, D::Label)> for MnistDataset<D>
where
    D: DataSet + Send + Sync,
    D::Image: Send + Sync + Copy,
    D::Label: Send + Sync + Copy,
{
    fn get(&self, i: usize) -> Option<(D::Image, D::Label)> {
        self.items.get(i).copied()
    }
    fn len(&self) -> usize {
        self.items.len()
    }
}
