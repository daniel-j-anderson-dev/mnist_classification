use activation_functions::{rectified_linear_unit, soft_max, SoftMax};

pub mod activation_functions;

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    layers: Box<[Layer]>,
    input_node_count: usize,
}
impl NeuralNetwork {
    /// # Panics
    /// if `layer_sizes` is empty or any of the layer sizes are 0
    pub fn new(layer_sizes: impl AsRef<[usize]>) -> Self {
        use rand::Rng;

        let layer_sizes = layer_sizes.as_ref();
        assert!(layer_sizes.len() != 0);
        assert!(layer_sizes.iter().any(|&layer_size| layer_size != 0));

        let mut rng = rand::thread_rng();

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

        NeuralNetwork {
            layers,
            input_node_count: layer_sizes[0],
        }
    }

    pub fn forward_propagation(&mut self, inputs: impl AsRef<[f64]>) {
        let mut activations = inputs.as_ref();
        assert!(activations.len() == self.input_node_count);

        for layer in self.layers.iter_mut() {
            for (node_index, (weights, bias)) in layer.weights.iter().zip(&layer.biases).enumerate()
            {
                let weighted_sum: f64 = weights
                    .iter()
                    .zip(activations)
                    .map(|(weight, activation)| weight * activation)
                    .sum();
                layer.activations[node_index] = rectified_linear_unit(weighted_sum + bias);
            }

            // pass activations to the next layer
            activations = &layer.activations;
        }

        self.layers
            .last_mut()
            .expect("NeuralNetwork::new would panic if there isn't at least 1 layer")
            .activations
            .soft_max();
    }

    pub fn backward_propagation(&mut self) {
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
    use crate::{
        dataset::IMAGE_SIZE,
        neural_network::{Layer, NeuralNetwork},
    };

    #[test]
    fn f() {
        let nn = NeuralNetwork::new([IMAGE_SIZE, 10, 10]);
        for l in nn.layers.iter() {
            println!("{:?}", l);
        }
    }
}
