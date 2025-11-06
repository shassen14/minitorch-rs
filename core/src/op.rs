//! Defines the `Op` trait for operations in the computational graph.
use crate::backend::Backend;
use dyn_clone::DynClone;
use std::fmt::Debug;

// This allows us to clone Box<dyn Op<B>>.
pub trait Op<B: Backend>: Debug + DynClone {
    fn backward(
        &self,
        upstream_grad: &B::TensorData,
        inputs: &[&B::TensorData],
    ) -> Vec<B::TensorData>;
}

dyn_clone::clone_trait_object!(<B: Backend> Op<B>);

// --- Add Operation ---
#[derive(Debug, Clone)]
pub struct Add;

impl<B: Backend> Op<B> for Add {
    fn backward(
        &self,
        upstream_grad: &B::TensorData,
        _inputs: &[&B::TensorData],
    ) -> Vec<B::TensorData> {
        vec![upstream_grad.clone(), upstream_grad.clone()]
    }
}

// --- Multiply Operation ---
#[derive(Debug, Clone)]
pub struct Mul;

impl<B: Backend> Op<B> for Mul {
    fn backward(
        &self,
        upstream_grad: &B::TensorData,
        inputs: &[&B::TensorData],
    ) -> Vec<B::TensorData> {
        // The chain rule for multiplication `c = a * b` is:
        // dL/da = dL/dc * dc/da = upstream_grad * b
        // dL/db = dL/dc * dc/db = upstream_grad * a
        let a_data = inputs[0];
        let b_data = inputs[1];

        // We need a `mul` operation on our backend.
        let backend = B::default();
        let grad_a = backend.mul(upstream_grad, b_data);
        let grad_b = backend.mul(upstream_grad, a_data);

        vec![grad_a, grad_b]
    }
}
