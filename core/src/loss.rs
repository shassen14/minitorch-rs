//! Defines common loss functions.
use crate::backend::Backend;
use crate::tensor::{Tensor, mean, pow, sub};

pub fn mse_loss<B: Backend + Default>(prediction: &Tensor<B>, target: &Tensor<B>) -> Tensor<B> {
    // Formula: L = mean((prediction - target)^2)
    let diff = sub(prediction, target);
    let squared_diff = pow(&diff, 2.0);
    let loss = mean(&squared_diff);
    loss
}
