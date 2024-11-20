pub mod dataset;
pub mod neural_network;

pub fn one_hot_encode(label: usize) -> [f64; 10] {
    assert!(label < 10, "there are only ten digits");
    let mut encoded = [0.0; 10];
    encoded[label] = 1.0;
    encoded
}
