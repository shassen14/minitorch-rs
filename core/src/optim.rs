//! Defines the optimizer, which updates a model's parameters based on gradients.
use crate::backend::Backend;
use crate::tensor::Tensor;

// --- The Optimizer Trait ---
// Any optimizer will need to be able to step and zero the gradients.
pub trait Optimizer<B: Backend> {
    fn step(&mut self);
    fn zero_grad(&mut self);
}

// --- Stochastic Gradient Descent (SGD) ---
pub struct Sgd<B: Backend + Default> {
    parameters: Vec<Tensor<B>>,
    learning_rate: f32,
}

impl<B: Backend + Default> Sgd<B> {
    pub fn new(parameters: Vec<Tensor<B>>, learning_rate: f32) -> Self {
        Self {
            parameters,
            learning_rate,
        }
    }
}

impl<B: Backend + Default> Optimizer<B> for Sgd<B>
where
    B::TensorData: Clone,
{
    fn zero_grad(&mut self) {
        // Clear the gradients for all parameters.
        for p in &self.parameters {
            p.inner.borrow_mut().grad = None;
        }
    }

    fn step(&mut self) {
        let backend = B::default();
        let lr_tensor_data = backend.from_slice(&[self.learning_rate], &[1]);

        // Update each parameter using the formula:
        // param.data = param.data - learning_rate * param.grad
        for p in &self.parameters {
            let mut p_inner = p.inner.borrow_mut();
            if let Some(grad) = &p_inner.grad {
                // In-place subtraction would be more efficient, but let's use our
                // existing `sub` primitive for now.
                let scaled_grad = backend.mul(&lr_tensor_data, grad);
                p_inner.data = backend.sub(&p_inner.data, &scaled_grad);
            }
        }
    }
}
