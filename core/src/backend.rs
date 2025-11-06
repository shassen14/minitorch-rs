//! Defines the `Backend` trait, which abstracts over the computational engine.
use ndarray::{ArrayD, IxDyn};

pub trait Backend: Default + Sized {
    type TensorData;

    fn add(&self, a: &Self::TensorData, b: &Self::TensorData) -> Self::TensorData;

    fn from_slice(&self, data: &[f32], shape: &[usize]) -> Self::TensorData;

    fn to_vec(&self, tensor: &Self::TensorData) -> Vec<f32>;

    fn shape<'a>(&self, tensor: &'a Self::TensorData) -> &'a [usize];
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NdArrayBackend;

impl Backend for NdArrayBackend {
    // Note: For simplicity, this framework is currently hardcoded to use `f32`
    // as its primary data type, which is the standard in most deep learning frameworks.
    // A more advanced implementation would make the element type generic.
    type TensorData = ArrayD<f32>;

    fn add(&self, a: &Self::TensorData, b: &Self::TensorData) -> Self::TensorData {
        a + b
    }

    fn from_slice(&self, data: &[f32], shape: &[usize]) -> Self::TensorData {
        ArrayD::from_shape_vec(IxDyn(shape), data.to_vec()).unwrap()
    }

    fn to_vec(&self, tensor: &Self::TensorData) -> Vec<f32> {
        tensor.as_slice().unwrap().to_vec()
    }

    fn shape<'a>(&self, tensor: &'a Self::TensorData) -> &'a [usize] {
        tensor.shape()
    }
}
