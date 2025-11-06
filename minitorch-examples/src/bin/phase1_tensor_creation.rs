//! Phase 1: Demonstrates the creation of Tensors and basic arithmetic.
//!
//! This example shows the initial functionality of the Tensor struct:
//! - Creating a tensor from raw data.
//! - Performing a simple, element-wise addition.
//! - Using the overloaded `+` operator.

use core::backend::NdArrayBackend;
use core::tensor::Tensor;

fn main() {
    println!("--- MiniTorch-rs: Phase 1 - Tensor Creation & Ops ---");

    // Define the type alias for our CPU-based tensor for clarity.
    type MyTensor = Tensor<NdArrayBackend>;

    // Create two tensors from slice data.
    let a = MyTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]); // A 2x2 matrix
    let b = MyTensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]); // A 2x2 matrix

    println!("\nTensor a:\n{:#?}", a);
    println!("\nTensor b:\n{:#?}", b);

    // Perform addition using the overloaded `+` operator.
    let c = &a + &b;

    println!("\nResult of a + b:\n{:#?}", c);

    // --- Verification ---
    // We can borrow the inner data to check the results.
    let c_inner = c.inner.borrow();
    let expected_data = ndarray::arr2(&[[6.0, 8.0], [10.0, 12.0]]).into_dyn();

    // Compare the raw data.
    assert_eq!(c_inner.data, expected_data);
    println!("\nVerification successful: The addition result is correct.");
}

/*
Expected Output:
----------------
--- MiniTorch-rs: Phase 1 - Tensor Creation & Ops ---

Tensor a:
Tensor {
    shape: [
        2,
        2,
    ],
    data: [[1.0, 2.0],
           [3.0, 4.0]], shape=[2, 2], strides=[2, 1], layout=CFcf (0xf), dynamic ndim=2,
    grad: None,
    _children_count: 0,
    _op: None,
}

Tensor b:
Tensor {
    shape: [
        2,
        2,
    ],
    data: [[5.0, 6.0],
           [7.0, 8.0]], shape=[2, 2], strides=[2, 1], layout=CFcf (0xf), dynamic ndim=2,
    grad: None,
    _children_count: 0,
    _op: None,
}

Result of a + b:
Tensor {
    shape: [
        2,
        2,
    ],
    data: [[6.0, 8.0],
           [10.0, 12.0]], shape=[2, 2], strides=[2, 1], layout=CFcf (0xf), dynamic ndim=2,
    grad: None,
    _children_count: 2,
    _op: Some(
        Add,
    ),
}

Verification successful: The addition result is correct.
----------------
*/
