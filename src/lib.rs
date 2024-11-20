pub mod dataset;
pub mod neural_network;

pub fn rectified_linear_unit(x: f64) -> f64 {
    x.max(0.0)
}
