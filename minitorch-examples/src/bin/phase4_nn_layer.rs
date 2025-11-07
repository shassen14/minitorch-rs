//! Phase 4: Demonstrates building and using a neural network layer.
use core::backend::NdArrayBackend;
use core::nn::{Linear, Module};
use core::tensor::Tensor;

fn main() {
    println!("--- MiniTorch-rs: Phase 4 - Neural Network Layer Test ---");
    type MyTensor = Tensor<NdArrayBackend>;

    // Create a Linear layer: 3 input features, 2 output features
    let linear_layer = Linear::<NdArrayBackend>::new(3, 2, true);

    // Create a batch of 1 input vector
    let input = MyTensor::new(&[1.0, 2.0, 3.0], &[1, 3]);

    println!("\nInput:\n{:#?}", input);
    println!("\nLinear Layer Weights:\n{:#?}", linear_layer.weight);
    println!("\nLinear Layer Bias:\n{:#?}", linear_layer.bias);

    // Perform the forward pass
    let output = linear_layer.forward(&input);

    println!("\nOutput of forward pass:\n{:#?}", output);

    // Check the parameters
    let params = linear_layer.parameters();
    assert_eq!(params.len(), 2);
    println!("\nSuccessfully retrieved layer parameters.");
}
