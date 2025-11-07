//! Phase 4: Demonstrates a backpropagation pass through a MatMul operation.

use core::backend::NdArrayBackend;
use core::tensor::{Tensor, matmul};
use ndarray::arr2;

fn main() {
    println!("--- MiniTorch-rs: Phase 4 - MatMul Gradient Test ---");

    type MyTensor = Tensor<NdArrayBackend>;

    // A = [[1, 2], [3, 4]]
    let a = MyTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    // B = [[5, 6], [7, 8]]
    let b = MyTensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

    // C = A @ B
    // We create `_c` to build the graph node, but we don't use its data directly.
    let _c = matmul(&a, &b);

    // Manually set the upstream gradient for C.
    // In a real network, this would be computed by the next layer's backward pass.
    // Let's assume the gradient is a matrix of ones for simplicity.
    _c.inner.borrow_mut().grad = Some(arr2(&[[1.0, 1.0], [1.0, 1.0]]).into_dyn());

    // Manually trigger the backward pass for this single node.
    let c_inner = _c.inner.borrow();
    let op = c_inner._op.as_ref().unwrap();
    let upstream_grad = c_inner.grad.as_ref().unwrap();

    // We need to get references to the data of the *children* of `_c`, which are `a` and `b`.
    // The previous code was slightly incorrect by re-borrowing the children. This is more direct.
    let children_data: Vec<_> = c_inner
        ._children
        .iter()
        .map(|t| t.inner.borrow().data.clone())
        .collect();
    let children_data_refs: Vec<_> = children_data.iter().collect();

    let local_grads = op.backward(upstream_grad, &children_data_refs);

    let grad_a = &local_grads[0];
    let grad_b = &local_grads[1];

    println!("\nUpstream Gradient for C (all ones):\n{:?}", upstream_grad);
    println!("\nCalculated Gradient for A:\n{:?}", grad_a);
    println!("\nCalculated Gradient for B:\n{:?}", grad_b);

    // --- Verification ---
    // grad(A) = grad(C) @ B.T = [[1, 1], [1, 1]] @ [[5, 7], [6, 8]] = [[11, 15], [11, 15]]
    let expected_grad_a = arr2(&[[11.0, 15.0], [11.0, 15.0]]).into_dyn();
    assert_eq!(*grad_a, expected_grad_a);

    // grad(B) = A.T @ grad(C) = [[1, 3], [2, 4]] @ [[1, 1], [1, 1]] = [[4, 4], [6, 6]]
    let expected_grad_b = arr2(&[[4.0, 4.0], [6.0, 6.0]]).into_dyn();
    assert_eq!(*grad_b, expected_grad_b);

    println!("\nVerification successful: MatMul gradients are correct!");
}
