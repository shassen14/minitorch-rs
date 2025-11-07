//! Defines the neural network abstractions like `Module` and `Linear` layers.
use crate::backend::Backend;
use crate::tensor::{Tensor, add, matmul, relu};
use rand::Rng;

// --- The Module Trait ---
// Any struct that is a "layer" or a "model" will implement this.
pub trait Module<B: Backend> {
    fn forward(&self, input: &Tensor<B>) -> Tensor<B>;

    /// Returns a `Vec` of all learnable `Tensor` parameters in this module.
    fn parameters(&self) -> Vec<Tensor<B>>;
}

/// A fully connected linear layer: `y = xA^T + b`.
#[derive(Debug)]
pub struct Linear<B: Backend + Default> {
    pub weight: Tensor<B>,
    pub bias: Option<Tensor<B>>,
}

impl<B: Backend + Default> Linear<B> {
    /// Creates a new `Linear` layer with weights initialized using Xavier uniform initialization.
    pub fn new(in_features: usize, out_features: usize, has_bias: bool) -> Self {
        // Xavier uniform initialization for weights
        let limit = (6.0 / (in_features + out_features) as f32).sqrt();
        let mut rng = rand::rng();

        let weight_shape = &[in_features, out_features];
        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|_| rng.random_range(-limit..limit))
            .collect();

        let weight = Tensor::new(&weight_data, weight_shape);

        let bias = if has_bias {
            let bias_data = vec![0.0; out_features];
            Some(Tensor::new(&bias_data, &[out_features]))
        } else {
            None
        };

        Self { weight, bias }
    }
}

// Implement the Module trait for our Linear layer
impl<B: Backend + Default> Module<B> for Linear<B> {
    fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        // Operation: output = input @ weight + bias
        let output = matmul(input, &self.weight);

        if let Some(bias) = &self.bias {
            add(&output, bias)
        } else {
            output
        }
    }
    fn parameters(&self) -> Vec<Tensor<B>> {
        let mut params = vec![self.weight.clone()];
        if let Some(bias) = &self.bias {
            params.push(bias.clone());
        }
        params
    }
}

// --- 3. Create the `ReLU` Module ---
#[derive(Debug, Clone)]
pub struct ReLU;

impl<B: Backend + Default> Module<B> for ReLU {
    fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        relu(input)
    }

    fn parameters(&self) -> Vec<Tensor<B>> {
        Vec::new() // ReLU has no learnable parameters
    }
}

// --- 4. Create the `Sequential` Module ---
pub struct Sequential<B: Backend> {
    layers: Vec<Box<dyn Module<B>>>,
}

impl<B: Backend + Default> Sequential<B> {
    pub fn new(layers: Vec<Box<dyn Module<B>>>) -> Self {
        Self { layers }
    }
}

impl<B: Backend + Default> Module<B> for Sequential<B> {
    fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        self.layers
            .iter()
            .fold(input.clone(), |current_input, layer| {
                layer.forward(&current_input)
            })
    }

    fn parameters(&self) -> Vec<Tensor<B>> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }
}
