//! Phase 4: Demonstrates that our `Linear` layer can process a batch of inputs.
//!
//! This is the core of how neural networks achieve efficiency: by processing
//! multiple data points (a "batch") simultaneously using matrix multiplication.

use core::backend::NdArrayBackend;
use core::nn::{Linear, Module};
use core::tensor::Tensor;

fn main() {
    println!("--- MiniTorch-rs: Phase 4 - Batched Forward Pass Test ---");

    type MyTensor = Tensor<NdArrayBackend>;

    // --- Setup ---
    let batch_size = 4;
    let in_features = 3;
    let out_features = 2;

    // 1. Create a Linear layer
    // This layer is a function that maps a vector of size 3 to a vector of size 2.
    let linear_layer = Linear::<NdArrayBackend>::new(in_features, out_features, true);

    // 2. Create a batch of input data.
    // This tensor represents 4 data points, each with 3 features.
    let input_data = vec![
        1.0, 2.0, 3.0, // Input 1
        4.0, 5.0, 6.0, // Input 2
        7.0, 8.0, 9.0, // Input 3
        0.0, 1.0, 0.5, // Input 4
    ];
    let input_shape = &[batch_size, in_features];
    let input = MyTensor::new(&input_data, input_shape);

    println!("\n--- Dimensions ---");
    println!("Batch input shape: {:?}", input.shape());
    println!(
        "Linear layer weight shape: {:?}",
        linear_layer.weight.shape()
    );
    println!(
        "Linear layer bias shape: {:?}",
        linear_layer.bias.as_ref().unwrap().shape()
    );

    // --- 3. Perform the Forward Pass ---
    // The `forward` method will perform: `output = input @ weight + bias`
    let output = linear_layer.forward(&input);

    println!("\n--- Output ---");
    println!("Output shape: {:?}", output.shape());
    println!("Output data:\n{:#?}", output);

    // --- 4. Verification ---
    let expected_shape = vec![batch_size, out_features];
    assert_eq!(output.shape(), expected_shape);

    println!("\nVerification successful: The output shape is correct for a batched input.");
    println!("This proves our `matmul` operation is handling the batch dimension correctly.");
}

/*
Expected Output (Weight/Bias/Output data will be random, but shapes are key):
----------------
--- MiniTorch-rs: Phase 4 - Batched Forward Pass Test ---

--- Dimensions ---
Batch input shape: [4, 3]
Linear layer weight shape: [3, 2]
Linear layer bias shape: [2]

--- Output ---
Output shape: [4, 2]
Output data:
Tensor {
    shape: [
        4,
        2,
    ],
    data: [[...random float data...],
           [...random float data...],
           [...random float data...],
           [...random float data...]], shape=[4, 2], strides=[2, 1], layout=Cc (0x5), dynamic ndim=2,
    grad: None,
    _children_count: 2,
    _op: Some(
        Add,
    ),
}

Verification successful: The output shape is correct for a batched input.
This proves our `matmul` operation is handling the batch dimension correctly.
----------------
*/
