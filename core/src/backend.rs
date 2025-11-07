//! Defines the `Backend` trait, which abstracts over the computational engine.
//!
//! This is the core of the framework's "backend-agnostic" design. The `Tensor`
//! and `Op` logic is written to be generic over any type that implements this
//! `Backend` trait. This allows us to swap out the underlying computation engine
//! (e.g., from a CPU engine to a GPU engine) without changing the high-level
//! autograd and neural network code.
//!
//! This is a form of **Policy-Based Design**.

use ndarray::{ArrayD, IxDyn, linalg::Dot};
use std::fmt::Debug;

/// The `Backend` trait defines the contract for a computational engine.
/// It specifies the types and primitive operations that the engine must provide.
//
// We add trait bounds here:
// - `Default`: Allows us to easily create a new instance of the backend.
// - `Sized`: A standard requirement for many generic types.
// - `Clone` + `Copy`: Since our `NdArrayBackend` is a zero-sized struct, it's cheap
//   to copy, and this makes it easier to pass around.
pub trait Backend: Default + Sized + Clone + Copy {
    /// `TensorData` is an "associated type". It defines a placeholder for the
    /// specific data structure that this backend will use to store tensor data.
    /// - For `NdArrayBackend`, this will be `ndarray::ArrayD<f32>`.
    /// - For a future `WgpuBackend`, this might be `wgpu::Buffer`.
    ///
    /// The `Clone` and `Debug` bounds mean that whatever type `TensorData` is,
    /// it must be cloneable and printable for debugging.
    type TensorData: Clone + Debug;

    // --- Primitive Mathematical Operations ---
    // The framework's `Op`s will delegate their actual computations to these methods.

    /// Performs element-wise addition of two tensors.
    fn add(&self, a: &Self::TensorData, b: &Self::TensorData) -> Self::TensorData;

    /// Performs element-wise multiplication of two tensors.
    fn mul(&self, a: &Self::TensorData, b: &Self::TensorData) -> Self::TensorData;

    /// Performs matrix multiplication.
    fn matmul(&self, a: &Self::TensorData, b: &Self::TensorData) -> Self::TensorData;

    /// Applies the Rectified Linear Unit (ReLU) activation function element-wise.
    fn relu(&self, a: &Self::TensorData) -> Self::TensorData;

    /// Returns a tensor of 1s where a > scalar, and 0s otherwise. Used for ReLU backward pass.
    fn gt_scalar(&self, a: &Self::TensorData, scalar: f32) -> Self::TensorData;

    /// Transposes a tensor by swapping two axes.
    fn transpose(&self, a: &Self::TensorData, axis1: usize, axis2: usize) -> Self::TensorData;

    // --- Data Management & Creation ---
    // These methods handle the lifecycle of tensor data.

    /// Creates a new tensor from a flat slice of `f32` data and a given shape.
    /// This is the primary way data gets from the "outside world" into our backend.
    fn from_slice(&self, data: &[f32], shape: &[usize]) -> Self::TensorData;

    /// Copies data from the backend's tensor representation back into a standard `Vec<f32>`.
    /// This is used for retrieving results or debugging.
    fn to_vec(&self, tensor: &Self::TensorData) -> Vec<f32>;

    /// Returns a slice representing the shape (dimensions) of the tensor data.
    /// The `'a` lifetime annotation ensures the returned slice is valid for as
    /// long as the `tensor` reference it's derived from.
    fn shape<'a>(&self, tensor: &'a Self::TensorData) -> &'a [usize];

    /// Creates a new tensor of a given shape, filled with zeros.
    /// Crucial for initializing gradients.
    fn zeros(&self, shape: &[usize]) -> Self::TensorData;

    /// Creates a new tensor of a given shape, filled with ones.
    /// Used to initialize the starting gradient in the `backward` pass.
    fn ones(&self, shape: &[usize]) -> Self::TensorData;
}

// ##################################################################
//  CPU Backend Implementation
// ##################################################################

/// `NdArrayBackend` is our concrete implementation of the `Backend` trait for CPU computation.
/// It's a "zero-sized struct" because it has no fields. Its only purpose is to be a
/// type on which we can implement the trait.
#[derive(Debug, Clone, Copy, Default)]
pub struct NdArrayBackend;

impl Backend for NdArrayBackend {
    // Note: For simplicity, this framework is currently hardcoded to use `f32`
    // as its primary data type, which is the standard in most deep learning frameworks.
    // A more advanced implementation would make the element type generic.
    type TensorData = ArrayD<f32>;

    /// `ndarray` overloads the `+` operator, so this is a simple one-liner.
    fn add(&self, a: &Self::TensorData, b: &Self::TensorData) -> Self::TensorData {
        a + b
    }

    /// `ndarray` overloads the `*` operator for element-wise multiplication.
    fn mul(&self, a: &Self::TensorData, b: &Self::TensorData) -> Self::TensorData {
        a * b
    }

    /// Performs matrix multiplication.
    fn matmul(&self, a: &Self::TensorData, b: &Self::TensorData) -> Self::TensorData {
        a.dot(b)
    }

    /// Applies the Rectified Linear Unit (ReLU) activation function element-wise.
    fn relu(&self, a: &Self::TensorData) -> Self::TensorData {
        a.mapv(|x| x.max(0.0)) // `mapv` applies a closure to each element.
    }

    /// Returns a tensor of 1s where a > scalar, and 0s otherwise. Used for ReLU backward pass.
    fn gt_scalar(&self, a: &Self::TensorData, scalar: f32) -> Self::TensorData {
        a.mapv(|x| if x > scalar { 1.0 } else { 0.0 })
    }

    /// Transposes a tensor by swapping two axes.
    fn transpose(&self, a: &Self::TensorData, axis1: usize, axis2: usize) -> Self::TensorData {
        let mut axes: Vec<_> = (0..a.ndim()).collect();
        axes.swap(axis1, axis2);
        a.clone().permuted_axes(axes)
    }

    /// `ndarray`'s constructor to create an array from the shape and vector data.
    fn from_slice(&self, data: &[f32], shape: &[usize]) -> Self::TensorData {
        ArrayD::from_shape_vec(IxDyn(shape), data.to_vec()).unwrap()
    }

    /// `ndarray` provides methods to convert the array back to a `Vec`.
    fn to_vec(&self, tensor: &Self::TensorData) -> Vec<f32> {
        tensor.as_slice().unwrap().to_vec()
    }

    /// `ndarray` arrays have a `.shape()` method that returns exactly what we need.
    fn shape<'a>(&self, tensor: &'a Self::TensorData) -> &'a [usize] {
        tensor.shape()
    }

    /// `ndarray` has a built-in constructor for an array of zeros.
    fn zeros(&self, shape: &[usize]) -> Self::TensorData {
        ArrayD::zeros(IxDyn(shape))
    }

    /// `ndarray` has a built-in constructor for an array of ones.
    fn ones(&self, shape: &[usize]) -> Self::TensorData {
        ArrayD::ones(IxDyn(shape))
    }
}
