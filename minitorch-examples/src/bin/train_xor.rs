//! Phase 5: Puts everything together to train a neural network on the XOR problem.
use core::backend::{Backend, NdArrayBackend};
use core::loss::mse_loss;
use core::nn::{Linear, Module, ReLU, Sequential}; // Import Sequential
use core::optim::{Optimizer, Sgd};
use core::tensor::Tensor;

fn main() {
    println!("--- MiniTorch-rs: Phase 5 - Training a Sequential MLP for XOR ---");
    type MyTensor = Tensor<NdArrayBackend>;
    type MyBackend = NdArrayBackend;

    // 1. Create the dataset
    let inputs = MyTensor::new(&[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2]);
    let targets = MyTensor::new(&[0.0, 1.0, 1.0, 0.0], &[4, 1]);

    // 2. Create the model using the Sequential container
    // This is much cleaner and more composable!
    let model = Sequential::<MyBackend>::new(vec![
        Box::new(Linear::new(2, 4, true)),
        Box::new(ReLU),
        Box::new(Linear::new(4, 1, true)),
    ]);

    // The optimizer gets the parameters from the sequential model automatically.
    let mut optimizer = Sgd::new(model.parameters(), 0.1);

    // 3. The Training Loop
    println!("\n--- Starting Training ---");
    for epoch in 1..=1000 {
        // Increased epochs for better convergence
        // The forward pass now uses the trait method.
        let predictions = model.forward(&inputs);
        let loss = mse_loss(&predictions, &targets);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if epoch % 100 == 0 || epoch == 1 {
            let loss_value = MyBackend::default().to_vec(&loss.inner.borrow().data)[0];
            println!("Epoch: {:>4}, Loss: {:.8}", epoch, loss_value);
        }
    }

    // 4. Final Predictions
    println!("\n--- Final Predictions ---");
    let final_predictions = model.forward(&inputs);
    let results = MyBackend::default().to_vec(&final_predictions.inner.borrow().data);

    println!("Input: [0, 0], Target: 0.0, Prediction: {:.4}", results[0]);
    println!("Input: [0, 1], Target: 1.0, Prediction: {:.4}", results[1]);
    println!("Input: [1, 0], Target: 1.0, Prediction: {:.4}", results[2]);
    println!("Input: [1, 1], Target: 0.0, Prediction: {:.4}", results[3]);

    let loss = mse_loss(&final_predictions, &targets);
    let final_loss = MyBackend::default().to_vec(&loss.inner.borrow().data)[0];
    println!("\nFinal Loss: {}", final_loss);
    assert!(final_loss < 0.01, "Loss should be very low after training.");
    println!("\nVerification successful: Model has learned XOR!");
}
