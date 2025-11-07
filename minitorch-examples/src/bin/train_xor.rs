//! Phase 5: Puts everything together to train a neural network on the XOR problem.
use core::backend::{Backend, NdArrayBackend};
use core::loss::mse_loss;
use core::nn::{Linear, Module};
use core::optim::{Optimizer, Sgd};
use core::tensor::{Tensor, relu};

// --- Define our Neural Network Model ---
// An MLP with one hidden layer: Linear -> ReLU -> Linear
struct Mlp<B: Backend + Default> {
    l1: Linear<B>,
    l2: Linear<B>,
}

impl<B: Backend + Default> Mlp<B> {
    fn new() -> Self {
        // 2 input features, 4 hidden features, 1 output feature
        Self {
            l1: Linear::new(2, 4, true),
            l2: Linear::new(4, 1, true),
        }
    }

    fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let x = self.l1.forward(input);
        let x = relu(&x);
        self.l2.forward(&x)
    }
}

// Implement the `Module` trait so the optimizer can find its parameters.
impl<B: Backend + Default> Module<B> for Mlp<B> {
    fn parameters(&self) -> Vec<Tensor<B>> {
        let mut params = self.l1.parameters();
        params.extend(self.l2.parameters());
        params
    }
}

fn main() {
    println!("--- MiniTorch-rs: Phase 5 - Training an MLP for XOR ---");
    type MyTensor = Tensor<NdArrayBackend>;

    // 1. Create the dataset
    let inputs = MyTensor::new(&[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2]);
    let targets = MyTensor::new(&[0.0, 1.0, 1.0, 0.0], &[4, 1]);

    // 2. Create the model and optimizer
    let model = Mlp::<NdArrayBackend>::new();
    let mut optimizer = Sgd::new(model.parameters(), 0.1); // Learning rate = 0.1

    // 3. The Training Loop
    println!("\n--- Starting Training ---");
    for epoch in 1..=1000 {
        // Forward pass: get the model's predictions
        let predictions = model.forward(&inputs);

        // Calculate the loss
        let loss = mse_loss(&predictions, &targets);

        // Backward pass: compute gradients
        optimizer.zero_grad(); // Reset gradients from previous epoch
        loss.backward();

        // Update the model's weights
        optimizer.step();

        if epoch % 100 == 0 || epoch == 1 {
            let loss_value = NdArrayBackend.to_vec(&loss.inner.borrow().data)[0];
            println!("Epoch: {}, Loss: {}", epoch, loss_value);
        }
    }

    // 4. Final Predictions (Inference)
    println!("\n--- Final Predictions ---");
    let final_predictions = model.forward(&inputs);
    let results = NdArrayBackend.to_vec(&final_predictions.inner.borrow().data);

    println!("Input: [0, 0], Target: 0.0, Prediction: {:.4}", results[0]);
    println!("Input: [0, 1], Target: 1.0, Prediction: {:.4}", results[1]);
    println!("Input: [1, 0], Target: 1.0, Prediction: {:.4}", results[2]);
    println!("Input: [1, 1], Target: 0.0, Prediction: {:.4}", results[3]);

    let loss = mse_loss(&final_predictions, &targets);
    let final_loss = NdArrayBackend.to_vec(&loss.inner.borrow().data)[0];
    println!("\nFinal Loss: {}", final_loss);
    assert!(final_loss < 0.01, "Loss should be very low after training.");
    println!("\nVerification successful: Model has learned XOR!");
}
