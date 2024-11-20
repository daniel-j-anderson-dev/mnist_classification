pub fn rectified_linear_unit(x: f64) -> f64 {
    x.max(0.0)
}
pub fn rectified_linear_unit_derivative(x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        1.0
    }
}

pub fn soft_max(mut z: impl AsMut<[f64]>) {
    let z = z.as_mut();

    let sum = z.iter().map(|z_i| z_i.exp()).sum::<f64>();

    for z_i in z.iter_mut() {
        *z_i /= sum;
    }
}

pub trait SoftMax {
    fn soft_max(&mut self);
}
impl<Arr> SoftMax for Arr
where
    Arr: AsMut<[f64]>,
{
    fn soft_max(&mut self) {
        soft_max(self);
    }
}
