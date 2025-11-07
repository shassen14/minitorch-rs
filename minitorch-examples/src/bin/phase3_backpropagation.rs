//! Phase 3: Demonstrates a simple backpropagation pass.
use core::backend::{Backend, NdArrayBackend};
use core::tensor::Tensor;

fn main() {
    println!("--- MiniTorch-rs: Phase 3 - Backpropagation Test ---");

    type MyTensor = Tensor<NdArrayBackend>;

    // Let's test a simple expression: d = (a * b) + c
    let a = MyTensor::new(&[2.0], &[1]);
    let b = MyTensor::new(&[3.0], &[1]);
    let c = MyTensor::new(&[4.0], &[1]);

    let e = &a + &b;
    let d = &e * &c;

    // --- Expected Gradients (Manual Calculation) ---
    // d = (a+b) * c
    // dd/da = c = 4.0
    // dd/db = c = 4.0
    // dd/dc = a + b = 5.0

    // Run backpropagation from the final node `d`.
    d.backward();

    println!("\nFinal Tensor d:\n{:#?}", d);
    println!("\n--- Gradients after backward() ---");
    println!("Gradient of a:\n{:#?}", a.inner.borrow().grad);
    println!("Gradient of b:\n{:#?}", b.inner.borrow().grad);
    println!("Gradient of c:\n{:#?}", c.inner.borrow().grad);

    // --- Verification ---
    let backend = NdArrayBackend::default();
    let a_grad = a.inner.borrow().grad.as_ref().unwrap().clone();
    let b_grad = b.inner.borrow().grad.as_ref().unwrap().clone();
    let c_grad = c.inner.borrow().grad.as_ref().unwrap().clone();

    assert_eq!(backend.to_vec(&a_grad), vec![4.0]);
    assert_eq!(backend.to_vec(&b_grad), vec![4.0]);
    assert_eq!(backend.to_vec(&c_grad), vec![5.0]);

    println!("\nVerification successful: Gradients are correct!");
}
