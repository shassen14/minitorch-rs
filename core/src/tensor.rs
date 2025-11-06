//! Defines the generic Tensor struct and its operations.
use crate::backend::Backend;
use crate::op::Op;
use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;

// `B::TensorData` is handled by the `Backend` trait definition.
// `Box<dyn Op<B>>` is handled by our `dyn-clone` setup in `op.rs`.
#[derive(Clone)]
pub struct TensorInner<B: Backend> {
    pub data: B::TensorData,
    pub grad: Option<B::TensorData>,
    pub _children: Vec<Tensor<B>>,
    pub _op: Option<Box<dyn Op<B>>>,
}

#[derive(Clone)]
pub struct Tensor<B: Backend> {
    pub inner: Rc<RefCell<TensorInner<B>>>,
}

impl<B: Backend + Default> Tensor<B> {
    pub fn new(data: &[f32], shape: &[usize]) -> Self {
        let backend = B::default();
        Tensor {
            inner: Rc::new(RefCell::new(TensorInner {
                data: backend.from_slice(data, shape),
                grad: None,
                _children: Vec::new(),
                _op: None,
            })),
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        let inner = self.inner.borrow();
        let backend = B::default();
        backend.shape(&inner.data).to_vec()
    }
}

impl<B: Backend + Default> Debug for Tensor<B>
where
    B::TensorData: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.borrow();
        let backend = B::default();
        f.debug_struct("Tensor")
            .field("shape", &backend.shape(&inner.data))
            .field("data", &inner.data)
            .field("grad", &inner.grad)
            .field("_children_count", &inner._children.len())
            .field("_op", &inner._op)
            .finish()
    }
}

// --- High-level `add` operation ---
pub fn add<B: Backend + Default>(a: &Tensor<B>, b: &Tensor<B>) -> Tensor<B> {
    let backend = B::default();
    let a_inner = a.inner.borrow();
    let b_inner = b.inner.borrow();

    let result_data = backend.add(&a_inner.data, &b_inner.data);

    Tensor {
        inner: Rc::new(RefCell::new(TensorInner {
            data: result_data,
            grad: None,
            _children: vec![a.clone(), b.clone()],
            _op: Some(Box::new(crate::op::Add)),
        })),
    }
}

impl<B: Backend + Default> std::ops::Add for &Tensor<B> {
    type Output = Tensor<B>;
    fn add(self, rhs: Self) -> Self::Output {
        add(self, rhs)
    }
}

// --- High-level `mul` operation ---
pub fn mul<B: Backend + Default>(a: &Tensor<B>, b: &Tensor<B>) -> Tensor<B> {
    let backend = B::default();
    let a_inner = a.inner.borrow();
    let b_inner = b.inner.borrow();

    // Forward pass
    let result_data = backend.mul(&a_inner.data, &b_inner.data);

    // Create the new tensor and store its graph history
    Tensor {
        inner: Rc::new(RefCell::new(TensorInner {
            data: result_data,
            grad: None,
            _children: vec![a.clone(), b.clone()],
            _op: Some(Box::new(crate::op::Mul)),
        })),
    }
}

// Operator overloading for `a * b`
impl<B: Backend + Default> std::ops::Mul for &Tensor<B> {
    type Output = Tensor<B>;
    fn mul(self, rhs: Self) -> Self::Output {
        mul(self, rhs)
    }
}
