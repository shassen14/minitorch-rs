//! Defines the `Op` (Operation) trait, which is the core of the autograd engine.
//!
//! Each mathematical operation (like Add, Mul, ReLU, etc.) in our framework
//! will be a struct that implements this `Op` trait. This trait provides the
//! "recipe" for how to perform backpropagation for that specific operation.

use crate::backend::Backend;
use dyn_clone::DynClone;
use std::fmt::Debug;

/// The `Op` trait defines the contract for any operation in the computational graph.
///
/// An `Op` is a stateless object that knows how to compute the gradients of its
/// inputs given the gradient from its output (the "upstream" gradient).
//
// Trait Bounds:
// - `B: Backend`: The operation is generic over a backend.
// - `Debug`: Allows us to print the operation for debugging the graph.
// - `DynClone`: A "super-trait" that allows us to clone `Box<dyn Op<B>>` trait
//   objects. This is necessary for the `TensorInner` struct to be cloneable.
pub trait Op<B: Backend>: Debug + DynClone {
    /// This is the core of the chain rule.
    ///
    /// It calculates the gradients of the loss with respect to the inputs of this
    /// operation.
    ///
    /// # Arguments
    /// * `upstream_grad`: The gradient flowing backwards from the operation's output tensor.
    ///                    (dL/dOutput)
    /// * `inputs`: The original input data tensors that were used in the forward pass.
    ///
    /// # Returns
    /// A `Vec` containing the gradients with respect to each of the inputs.
    /// (e.g., `vec![dL/dInput1, dL/dInput2, ...]`)
    fn backward(
        &self,
        upstream_grad: &B::TensorData,
        inputs: &[&B::TensorData],
    ) -> Vec<B::TensorData>;
}

// This line provides the necessary implementation to make `Box<dyn Op<B>>` cloneable.
dyn_clone::clone_trait_object!(<B: Backend> Op<B>);

// ##################################################################
//  Concrete Operation Implementations
// ##################################################################

// --- Add Operation ---
// A zero-sized struct to represent the addition operation.
#[derive(Debug, Clone)]
pub struct Add;

impl<B: Backend> Op<B> for Add {
    // --- Chain Rule for Addition ---
    // If `C = A + B`, then:
    //   - The derivative of C with respect to A (dC/dA) is 1.
    //   - The derivative of C with respect to B (dC/dB) is 1.
    //
    // Using the chain rule, the gradient of the final Loss L is:
    //   - dL/dA = dL/dC * dC/dA = upstream_grad * 1.0 = upstream_grad
    //   - dL/dB = dL/dC * dC/dB = upstream_grad * 1.0 = upstream_grad
    //
    // So, the gradient is simply passed through to both inputs.
    fn backward(
        &self,
        upstream_grad: &B::TensorData,
        _inputs: &[&B::TensorData],
    ) -> Vec<B::TensorData> {
        vec![upstream_grad.clone(), upstream_grad.clone()]
    }
}

// --- Subtract Operation ---
#[derive(Debug, Clone)]
pub struct Sub;
impl<B: Backend> Op<B> for Sub {
    fn backward(
        &self,
        upstream_grad: &B::TensorData,
        _inputs: &[&B::TensorData],
    ) -> Vec<B::TensorData> {
        // dL/da = dL/dc * dc/da = upstream_grad * 1.0
        // dL/db = dL/dc * dc/db = upstream_grad * -1.0
        let backend = B::default();
        // Create a scalar tensor of -1
        let neg_one = backend.from_slice(&[-1.0], &[1]);
        let grad_b = backend.mul(upstream_grad, &neg_one);
        vec![upstream_grad.clone(), grad_b]
    }
}

// --- Multiply Operation ---
// A zero-sized struct to represent element-wise multiplication.
#[derive(Debug, Clone)]
pub struct Mul;

impl<B: Backend> Op<B> for Mul {
    fn backward(
        &self,
        upstream_grad: &B::TensorData,
        inputs: &[&B::TensorData],
    ) -> Vec<B::TensorData> {
        // --- Chain Rule for Multiplication ---
        // If `C = A * B`, then:
        //   - The derivative of C with respect to A (dC/dA) is B.
        //   - The derivative of C with respect to B (dC/dB) is A.
        //
        // Using the chain rule, the gradient of the final Loss L is:
        //   - dL/dA = dL/dC * dC/dA = upstream_grad * B
        //   - dL/dB = dL/dC * dC/dB = upstream_grad * A
        let a_data = inputs[0];
        let b_data = inputs[1];

        let backend = B::default();
        let grad_a = backend.mul(upstream_grad, b_data);
        let grad_b = backend.mul(upstream_grad, a_data);

        vec![grad_a, grad_b]
    }
}

#[derive(Debug, Clone)]
pub struct Pow {
    pub power: f32,
}
impl<B: Backend> Op<B> for Pow {
    fn backward(
        &self,
        upstream_grad: &B::TensorData,
        inputs: &[&B::TensorData],
    ) -> Vec<B::TensorData> {
        // For y = x^n, dy/dx = n * x^(n-1)
        // dL/dx = dL/dy * dy/dx = upstream_grad * (n * x^(n-1))
        let base = inputs[0];
        let backend = B::default();

        let n = backend.from_slice(&[self.power], &[1]);
        let n_minus_1 = backend.from_slice(&[self.power - 1.0], &[1]);

        let pow_term = backend.pow(base, &n_minus_1);
        let factor = backend.mul(&n, &pow_term);
        let grad = backend.mul(upstream_grad, &factor);
        vec![grad]
    }
}

#[derive(Debug, Clone)]
pub struct Mean;
impl<B: Backend> Op<B> for Mean {
    fn backward(
        &self,
        upstream_grad: &B::TensorData,
        inputs: &[&B::TensorData],
    ) -> Vec<B::TensorData> {
        // If y = mean(x), dy/dx_i = 1/N for all i.
        // dL/dx_i = dL/dy * dy/dx_i = upstream_grad * (1/N)
        let input = inputs[0];
        let backend = B::default();
        let n = backend.shape(input).iter().product::<usize>() as f32;
        let n_inv = backend.from_slice(&[1.0 / n], &[1]);
        let grad = backend.mul(upstream_grad, &n_inv);
        // This gradient needs to be broadcasted to the shape of the input.
        let broadcasted_grad = backend.expand(&grad, backend.shape(input));
        vec![broadcasted_grad]
    }
}

// --- ReLU Operation ---
// A zero-sized struct to represent ReLU.
#[derive(Debug, Clone)]
pub struct ReLU;

impl<B: Backend> Op<B> for ReLU {
    fn backward(
        &self,
        upstream_grad: &B::TensorData,
        inputs: &[&B::TensorData],
    ) -> Vec<B::TensorData> {
        let input_data = inputs[0];
        let backend = B::default();

        let mask = backend.gt_scalar(input_data, 0.0);
        let grad_a = backend.mul(upstream_grad, &mask);

        vec![grad_a]
    }
}

// --- Matrix Multiplication Operation ---
#[derive(Debug, Clone)]
pub struct MatMul;
impl<B: Backend> Op<B> for MatMul {
    fn backward(
        &self,
        upstream_grad: &B::TensorData,
        inputs: &[&B::TensorData],
    ) -> Vec<B::TensorData> {
        // Let the forward pass be `C = A @ B`.
        // The chain rule gives us the gradients with respect to the inputs A and B:
        // grad(A) = grad(C) @ B.T
        // grad(B) = A.T @ grad(C)

        let a_data = inputs[0];
        let b_data = inputs[1];
        let backend = B::default();

        let b_shape = backend.shape(b_data);
        let b_transposed = backend.transpose(b_data, b_shape.len() - 2, b_shape.len() - 1);
        let grad_a = backend.matmul(upstream_grad, &b_transposed);

        let a_shape = backend.shape(a_data);
        let a_transposed = backend.transpose(a_data, a_shape.len() - 2, a_shape.len() - 1);
        let grad_b = backend.matmul(&a_transposed, upstream_grad);

        vec![grad_a, grad_b]
    }
}
