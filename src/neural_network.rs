#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    layers: Box<[Layer]>,
}
impl NeuralNetwork {
    pub fn new(layer_sizes: impl AsRef<[usize]>) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let layer_sizes = layer_sizes.as_ref();

        let layers = (0..layer_sizes.len())
            .map(|layer_index| {
                if layer_index == 0 {
                    Layer {
                        number_of_nodes: layer_sizes[layer_index],
                        weights: Box::default(),
                        biases: Box::default(),
                        activations: Box::default(),
                    }
                } else {
                    Layer {
                        number_of_nodes: layer_sizes[layer_index],
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
                }
            })
            .collect();

        NeuralNetwork { layers }
    }
    pub fn forward_pass(&mut self) {
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
    number_of_nodes: usize,
}

