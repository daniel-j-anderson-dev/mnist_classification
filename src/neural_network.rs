use activation_functions::{rectified_linear_unit, rectified_linear_unit_derivative, SoftMax};

pub mod activation_functions;

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    layers: Box<[Layer]>,
    input_neuron_count: usize,
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
                    // generate a random weight for every combination of neuron indexes between the current row and the previous row
                    weights: (0..layer_sizes[layer_index])
                        .map(|_current_layer_neuron_index| {
                            (0..layer_sizes[layer_index - 1])
                                .map(|_previous_layer_neuron_index| rng.gen_range(-1.0..1.0))
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
            input_neuron_count: layer_sizes[0],
        }
    }

    fn last_layer(&self) -> &Layer {
        self.layers
            .last()
            .expect("NeuralNetwork::new would panic if there isn't at least 1 layer")
    }
    fn last_layer_mut(&mut self) -> &mut Layer {
        self.layers
            .last_mut()
            .expect("NeuralNetwork::new would panic if there isn't at least 1 layer")
    }

    pub fn forward_propagation(&mut self, inputs: impl AsRef<[f64]>) {
        let mut previous_layer_activations = inputs.as_ref();
        assert!(
            previous_layer_activations.len() == self.input_neuron_count,
            "there must be exactly self.input_neuron_count inputs"
        );

        for layer in self.layers.iter_mut() {
            for (neuron_index, (weights, bias)) in
                layer.weights.iter().zip(&layer.biases).enumerate()
            {
                let weighted_sum: f64 = weights
                    .iter()
                    .zip(previous_layer_activations)
                    .map(|(weight, activation)| weight * activation)
                    .sum();
                layer.activations[neuron_index] = rectified_linear_unit(weighted_sum + bias);
            }

            // pass activations to the next layer
            previous_layer_activations = &layer.activations;
        }

        self.last_layer_mut().activations.soft_max();
    }

    pub fn backward_propagation(&mut self, targets: impl AsRef<[f64]>, learning_rate: f64) {
        let targets = targets.as_ref();

        assert_eq!(
            targets.len(),
            self.last_layer().number_of_neurons(),
            "the number of targets must be equal to the number of neurons in the final layer"
        );

        // compute error
        let mut deltas = self
            .last_layer()
            .activations
            .iter()
            .zip(targets)
            .map(|(output, target)| output - target)
            .collect::<Box<_>>();

        // back propagate through each layer
        for layer_index in (0..self.layers.len()).rev() {
            {
                let layer = &mut self.layers[layer_index];

                // calculate weights and biases gradients
                let weight_gradients = deltas
                    .iter()
                    .map(|delta| {
                        layer
                            .activations
                            .iter()
                            .map(|activation| delta * activation)
                            .collect::<Box<_>>()
                    })
                    .collect::<Box<_>>();

                let bias_gradients = deltas.clone();

                // update weights and biases according to their gradients
                for (weights, weight_gradients) in layer.weights.iter_mut().zip(weight_gradients) {
                    for (weight, weight_gradient) in weights.iter_mut().zip(weight_gradients) {
                        *weight -= learning_rate * weight_gradient;
                    }
                }

                for (bias, bias_gradient) in layer.biases.iter_mut().zip(bias_gradients) {
                    *bias -= learning_rate * bias_gradient;
                }
            }

            // calculate error for previous layer
            if layer_index > 0 {
                let layer = &self.layers[layer_index];
                let previous_layer = &self.layers[layer_index - 1];

                deltas = (0..previous_layer.number_of_neurons())
                    .map(|previous_layer_neuron_index| {
                        layer
                            .weights
                            .iter()
                            .zip(&deltas)
                            .fold(0.0, |sum, (weights, delta)| {
                                sum + weights[previous_layer_neuron_index] * delta
                            })
                            * rectified_linear_unit_derivative(
                                previous_layer.activations[previous_layer_neuron_index],
                            )
                    })
                    .collect();
            }
        }
    }
    
    pub fn train<O1, I1, O2, I2>(
        &mut self,
        inputs: O1,
        expected_outputs: O2,
        epochs: usize,
        learning_rate: f64,
    ) where
        O1: AsRef<[I1]>,
        I1: AsRef<[f64]>,
        O2: AsRef<[I2]>,
        I2: AsRef<[f64]>,
    {
        let inputs = inputs.as_ref();
        let expected_outputs = expected_outputs.as_ref();

        assert_eq!(
            inputs.len(),
            expected_outputs.len(),
            "Must have the same number of inputs and expected outputs"
        );

        for epoch_index in 0..epochs {
            let mut total_loss = 0.0;

            for (input, expected_output) in inputs
                .iter()
                .zip(expected_outputs)
                .map(|(i, eo)| (i.as_ref(), eo.as_ref()))
            {
                self.forward_propagation(input);

                // calculate loss
                let actual_output = &self.last_layer().activations;
                assert_eq!(
                    actual_output.len(),
                    expected_outputs.len(),
                    "expected output and actual output have different sizes"
                );

                let sample_loss = actual_output
                    .iter()
                    .zip(expected_output)
                    .map(|(actual, expected)| (actual - expected).powi(2))
                    .sum::<f64>();
                total_loss += sample_loss / expected_output.len() as f64;

                self.backward_propagation(expected_output, learning_rate);
            }

            println!(
                "Epoch {}/{} Loss: {}",
                epoch_index,
                epoch_index + 1,
                total_loss / inputs.len() as f64
            )
        }
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
    pub const fn number_of_neurons(&self) -> usize {
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
