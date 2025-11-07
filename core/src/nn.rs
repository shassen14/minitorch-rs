//! Defines the neural network abstractions like `Module` and `Linear` layers.
use crate::backend::Backend;
use crate::tensor::{Tensor, add, matmul}; // We'll add matmul soon
use rand::Rng;

// --- The Module Trait ---
// Any struct that is a "layer" or a "model" will implement this.
pub trait Module<B: Backend> {
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

        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|_| rng.random_range(-limit..limit))
            .collect();
        let weight = Tensor::new(&weight_data, &[in_features, out_features]);

        let bias = if has_bias {
            let bias_data = vec![0.0; out_features];
            Some(Tensor::new(&bias_data, &[out_features]))
        } else {
            None
        };

        Self { weight, bias }
    }

    /// Performs the forward pass of the linear layer.
    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        // Operation: output = input @ weight + bias
        let output = matmul(input, &self.weight); // We need to create the matmul op
        if let Some(bias) = &self.bias {
            add(&output, bias)
        } else {
            output
        }
    }
}

// Implement the Module trait for our Linear layer
impl<B: Backend + Default> Module<B> for Linear<B> {
    fn parameters(&self) -> Vec<Tensor<B>> {
        let mut params = vec![self.weight.clone()];
        if let Some(bias) = &self.bias {
            params.push(bias.clone());
        }
        params
    }
}
