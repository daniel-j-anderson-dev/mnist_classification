pub mod activation_functions;

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    layers: Box<[Layer]>,
}
impl NeuralNetwork {
    pub fn new(layer_sizes: impl AsRef<[usize]>) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let layer_sizes = layer_sizes.as_ref();

        let layers = (1..layer_sizes.len())
            .map(|layer_index| {
                Layer {
                    // generate a random weight for every combination of node indexes between the current row and the previous row
                    weights: (0..layer_sizes[layer_index])
                        .map(|_current_layer_node_index| {
                            (0..layer_sizes[layer_index - 1])
                                .map(|_previous_layer_node_index| rng.gen_range(-1.0..1.0))
                                .collect()
                        })
                        .collect(),

                    // initialize biases to 0.0
                    biases: std::iter::repeat(0.0)
                        .take(layer_sizes[layer_index])
                        .collect(),

                    // initialize activations to 0.0 since no passes have been made yet
                    activations: std::iter::repeat(0.0)
                        .take(layer_sizes[layer_index])
                        .collect(),
                }
            })
            .collect();

        NeuralNetwork { layers }
    }
    pub fn forward_pass(&mut self, inputs: impl AsRef<[f64]>) {
        unimplemented!()
    }
    pub fn backward_pass(&mut self) {
        unimplemented!()
    }
}

#[derive(Debug, Clone)]
pub struct Layer {
    /// adjacency matrix for this [Layer] and the previous.
    weights: Box<[Box<[f64]>]>,
    biases: Box<[f64]>,
    activations: Box<[f64]>,
}
impl Layer {
    pub const fn number_of_nodes(&self) -> usize {
        self.activations.len()
    }
}

#[cfg(test)]
mod test {
    use crate::neural_network::{Layer, NeuralNetwork};
}
