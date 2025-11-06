use core::backend::NdArrayBackend;
use core::tensor::Tensor;

fn main() {
    println!("--- MiniTorch-rs: Phase 2 - Computational Graph Test ---");

    type MyTensor = Tensor<NdArrayBackend>;

    let a = MyTensor::new(&[1.0, 2.0, 3.0], &[3]);
    let b = MyTensor::new(&[4.0, 5.0, 6.0], &[3]);
    let c = MyTensor::new(&[1.0, 2.0, 3.0], &[3]);

    println!("\nTensor a:\n{:#?}", a);
    println!("\nTensor b:\n{:#?}", b);

    // Perform a more complex operation: d = (a + b) * c
    let e = &a + &b;
    let d = &e * &c;

    println!("\nTensor a: {:?}", a);
    println!("Tensor b: {:?}", b);
    println!("Tensor c: {:?}", c);
    println!("\nIntermediate Tensor e = a + b:\n{:#?}", e);
    println!("\nFinal Tensor d = e * c:\n{:#?}", d);

    // --- Verification ---
    let d_inner = d.inner.borrow();
    assert_eq!(d_inner._children.len(), 2); // `d` was made from `e` and `c`
    assert!(d_inner._op.is_some());
    println!("\nVerification successful: Graph structure for `d` is correct.");

    let e_inner = d_inner._children[0].inner.borrow();
    assert_eq!(e_inner._children.len(), 2); // `e` was made from `a` and `b`
    println!("Verification successful: Graph structure for `e` is also correct.");
}
