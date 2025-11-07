//! Phase 4: Demonstrates the Mean Squared Error (MSE) loss function.
use core::backend::{Backend, NdArrayBackend};
use core::loss::mse_loss;
use core::tensor::Tensor;

fn main() {
    println!("--- MiniTorch-rs: Phase 4 - MSE Loss Test ---");
    type MyTensor = Tensor<NdArrayBackend>;

    let predictions = MyTensor::new(&[0.9, 0.2, 0.8, 0.1], &[4, 1]);
    let targets = MyTensor::new(&[1.0, 0.0, 1.0, 0.0], &[4, 1]);

    // Calculate the loss
    let loss = mse_loss(&predictions, &targets);

    println!("\nPredictions:\n{:#?}", predictions);
    println!("\nTargets:\n{:#?}", targets);
    println!("\nCalculated MSE Loss:\n{:#?}", loss);

    // --- Verification ---
    // diff = [-0.1, 0.2, -0.2, 0.1]
    // squared_diff = [0.01, 0.04, 0.04, 0.01]
    // mean = (0.01 + 0.04 + 0.04 + 0.01) / 4 = 0.1 / 4 = 0.025
    let loss_value = NdArrayBackend.to_vec(&loss.inner.borrow().data)[0];
    assert!((loss_value - 0.025).abs() < 1e-6);
    println!("\nVerification successful: Loss calculation is correct.");

    // --- Test the backward pass ---
    loss.backward();

    println!("\n--- Gradients after backward() ---");
    println!(
        "Gradient of predictions:\n{:#?}",
        predictions.inner.borrow().grad
    );

    // --- Verification of Gradients ---
    // dL/d_pred_i = (2/N) * (pred_i - target_i)
    // N = 4, so factor is 0.5
    // grad[0] = 0.5 * (0.9 - 1.0) = -0.05
    // grad[1] = 0.5 * (0.2 - 0.0) =  0.1
    // grad[2] = 0.5 * (0.8 - 1.0) = -0.1
    // grad[3] = 0.5 * (0.1 - 0.0) =  0.05
    let expected_grads = vec![-0.05, 0.1, -0.1, 0.05];
    let actual_grads = NdArrayBackend.to_vec(&predictions.inner.borrow().grad.as_ref().unwrap());

    for (expected, actual) in expected_grads.iter().zip(actual_grads.iter()) {
        assert!((expected - actual).abs() < 1e-6);
    }
    println!("\nVerification successful: Gradients for MSE loss are correct!");
}
