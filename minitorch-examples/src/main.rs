// We now use the specific backend we want.
use core::backend::NdArrayBackend;
use core::tensor::{Tensor, add};

fn main() {
    println!("--- MiniTorch-rs: Backend-Agnostic Test ---");

    // Create tensors using the backend.
    let a: Tensor<NdArrayBackend> = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let b: Tensor<NdArrayBackend> = Tensor::new(&[4.0, 5.0, 6.0], &[3]);

    println!("\nTensor a: {:?}", a);
    println!("Shape of a: {:?}", a.shape());

    println!("\nTensor b: {:?}", b);

    // Use our generic add function
    println!("\nResult of add(a, b):");
    let c = add(&a, &b);
    println!("{:?}", c);

    // Use the overloaded `+` operator
    println!("\nResult of a + b:");
    let d = &a + &b;
    println!("{:?}", d);
}
