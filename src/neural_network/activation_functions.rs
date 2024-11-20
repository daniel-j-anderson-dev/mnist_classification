pub fn rectified_linear_unit(mut z: impl AsMut<[f64]>) {
    for z_i in z.as_mut().iter_mut() {
        *z_i = z_i.max(0.0);
    }
}

pub fn soft_max(mut z: impl AsMut<[f64]>) {
    let z = z.as_mut();

    let sum = z.iter().map(|z_i| z_i.exp()).sum::<f64>();

    for z_i in z.iter_mut() {
        *z_i /= sum;
    }
}