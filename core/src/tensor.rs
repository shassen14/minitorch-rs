//! Defines the generic Tensor struct, the central data structure of the framework.

use crate::backend::Backend;
use crate::op::{MatMul, Mean, Op, Pow, ReLU, Sub};
use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

// ##################################################################
//  Core Data Structures
// ##################################################################

/// `TensorInner` holds the actual data and graph information.
///
/// This struct is wrapped in `Rc<RefCell<...>>` to create the `Tensor` type.
/// - `Rc`: Allows for multiple `Tensor` objects to share ownership of the same underlying data and graph node.
/// - `RefCell`: Provides "interior mutability," allowing us to modify the `grad` field
///   even when we only have an immutable reference to the `Tensor`.
//  NOTE: `Rc<RefCell<...>>` is for single threaded data structure.
//        For multi-threaded data, we use Arc<Mutex<...>>
#[derive(Clone)]
pub struct TensorInner<B: Backend> {
    /// The actual tensor data, stored in a backend-specific format.
    pub data: B::TensorData,
    /// The gradient of this tensor, computed during the backward pass.
    pub grad: Option<B::TensorData>,
    /// Pointers to the parent tensors that created this tensor in the forward pass.
    pub _children: Vec<Tensor<B>>,
    /// The operation that produced this tensor.
    pub _op: Option<Box<dyn Op<B>>>,
}

/// `Tensor` is the public-facing, lightweight handle to a node in the computational graph.
///
/// It's essentially a shared, mutable pointer (`Rc<RefCell<...>>`) to a `TensorInner`.
/// Cloning a `Tensor` is cheap, as it only increments a reference count.
#[derive(Clone)]
pub struct Tensor<B: Backend> {
    pub inner: Rc<RefCell<TensorInner<B>>>,
}

// To keep track of visited nodes during the topological sort, we need to
// be able to put Tensors in a HashSet. This requires implementing Eq and Hash.
// We can do this based on the memory address of the `Rc` pointer.
impl<B: Backend> PartialEq for Tensor<B> {
    fn eq(&self, other: &Self) -> bool {
        // Two tensors are considered the same node if they point to the same inner data.
        Rc::ptr_eq(&self.inner, &other.inner)
    }
}

impl<B: Backend> Eq for Tensor<B> {}

impl<B: Backend> Hash for Tensor<B> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // We hash the memory address of the pointer.
        Rc::as_ptr(&self.inner).hash(state);
    }
}

// ##################################################################
//  Tensor Implementation
// ##################################################################

impl<B: Backend + Default> Tensor<B>
where
    B::TensorData: Clone,
{
    /// Creates a new "leaf" tensor (a tensor with no history).
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

    /// Returns the shape of the tensor as a `Vec<usize>`.
    pub fn shape(&self) -> Vec<usize> {
        let inner = self.inner.borrow();
        let backend = B::default();
        backend.shape(&inner.data).to_vec()
    }

    // ==================================================================
    //  Backpropagation
    // ==================================================================
    //
    // Backpropagation is an algorithm that computes the chain rule of calculus
    // by traversing the computational graph in reverse topological order.
    //
    // --- How it Works: A Step-by-Step Trace ---
    //
    // Consider the expression: d = (a * b) + c
    // with initial values: a=2.0, b=3.0, c=4.0
    // The forward pass computes: e = a * b = 6.0, and d = e + c = 10.0
    //
    // The graph (with children pointing to operands) looks like this:
    //
    //        d (Add)
    //       /   \
    //      /     \
    //     e (Mul)  c
    //    /   \
    //   /     \
    //  a       b
    //
    // The goal is to find the gradients dd/da, dd/db, and dd/dc.
    //
    // 1. **Topological Sort:** Produces an ordered list of nodes.
    //    `sorted = [a, b, c, e, d]` (order of a,b,c can vary)
    //
    // 2. **Initialization:** The `backward()` pass starts from `d`. `d.grad` is set to 1.0.
    //
    // 3. **Reverse Traversal:** The algorithm iterates through `[d, c, e, b, a]`.
    //
    //    - **Process `d`:**
    //      - `op` is `Add`, `upstream_grad` is `d.grad` (1.0).
    //      - `Add.backward()` is called. The gradient for addition passes the upstream
    //        gradient through to both its children (`e` and `c`).
    //      - `e.grad` gets `1.0`. (This is dd/de)
    //      - `c.grad` gets `1.0`. (This is dd/dc - **Final gradient for c**)
    //
    //    - **Process `c`:**
    //      - Leaf node, no `_op`. Skip.
    //
    //    - **Process `e`:**
    //      - `op` is `Mul`, `upstream_grad` is `e.grad` (1.0).
    //      - `Mul.backward()` is called. Chain rule for `z = x*y` is `dz/dx=y`, `dz/dy=x`.
    //      - `a.grad` gets `e.grad * b.data` => `1.0 * 3.0 = 3.0` (**Final gradient for a**)
    //      - `b.grad` gets `e.grad * a.data` => `1.0 * 2.0 = 2.0` (**Final gradient for b**)
    //
    //    - **Process `b` and `a`:**
    //      - Leaf nodes, no `_op`. Skip.
    //
    // 4. **Final Gradients:** `a.grad`=3.0, `b.grad`=2.0, `c.grad`=1.0.
    //
    // This is the process this `backward()` method automates.
    //
    /// Performs backpropagation starting from this tensor..
    pub fn backward(&self) {
        // 1. Perform a topological sort of the graph
        // Get the standard "forward" dependency order.
        let sorted_nodes = self.topological_sort();

        // 2. Initialize the gradient of this starting tensor to ones.
        // This is the "dL/dL" = 1 starting point of the chain rule.
        let backend = B::default();
        let shape = self.shape();
        self.inner.borrow_mut().grad = Some(backend.ones(&shape));

        // 3. Propagate gradients backwards through the sorted list.
        // The reverse order ensures we process a node only after its "parents"
        // (the nodes that depend on it) have had their gradients accumulated.
        for node in sorted_nodes.iter().rev() {
            // We use `if let` to safely unwrap the gradient and operation.
            // Some nodes (like leaves) won't have an op or might not have a grad yet.
            if let (Some(grad), Some(op)) = (&node.inner.borrow().grad, &node.inner.borrow()._op) {
                // Collect references to the children's data
                let children_data: Vec<_> = node
                    .inner
                    .borrow()
                    ._children
                    .iter()
                    .map(|c| c.inner.borrow().data.clone())
                    .collect();

                // We need to convert from Vec<Ref<T>> to Vec<&T> for the `backward` call.
                let children_data_refs: Vec<_> = children_data.iter().collect();

                // Calculate the local gradients using the op's backward pass
                let local_grads = op.backward(grad, &children_data_refs);

                // Accumulate these new gradients into the children's `.grad` fields.
                for (i, child) in node.inner.borrow()._children.iter().enumerate() {
                    let mut child_inner = child.inner.borrow_mut();
                    let child_shape = backend.shape(&child_inner.data).to_vec();

                    if child_inner.grad.is_none() {
                        // Initialize the child's gradient to zeros if it's the first time
                        // we're backpropagating to it.
                        child_inner.grad = Some(backend.zeros(&child_shape));
                    }

                    // Add the new local gradient to the child's existing gradient.
                    if let Some(ref mut current_grad) = child_inner.grad {
                        *current_grad = backend.add(current_grad, &local_grads[i]);
                    }
                }
            }
        }
    }

    // --- Private Functions --

    /// Helper function to perform a topological sort using a depth-first search.
    /// This ensures that we get a linear ordering of nodes where every node
    /// appears before any of its parents.
    fn topological_sort(&self) -> Vec<Tensor<B>> {
        let mut sorted = Vec::new();
        let mut visited = HashSet::new();

        // This function performs a depth-first search to build the sorted list.
        self.build_topology(&mut sorted, &mut visited);

        sorted
    }

    /// The recursive part of the depth-first search for the topological sort.
    fn build_topology(&self, sorted: &mut Vec<Tensor<B>>, visited: &mut HashSet<Tensor<B>>) {
        // If we haven't visited this node yet
        if visited.insert(self.clone()) {
            for child in &self.inner.borrow()._children {
                // recursively visit all of its children first.
                child.build_topology(sorted, visited);
            }
            // After all children have been visited and added to the list, add this node.
            // This ensures a "post-order traversal", which is a valid topological sort.
            sorted.push(self.clone());
        }
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

pub fn relu<B: Backend + Default>(a: &Tensor<B>) -> Tensor<B> {
    let backend = B::default();
    let result_data = backend.relu(&a.inner.borrow().data);
    Tensor {
        inner: Rc::new(RefCell::new(TensorInner {
            data: result_data,
            grad: None,
            _children: vec![a.clone()],
            _op: Some(Box::new(ReLU)),
        })),
    }
}

pub fn matmul<B: Backend + Default>(a: &Tensor<B>, b: &Tensor<B>) -> Tensor<B> {
    let backend = B::default();
    let result_data = backend.matmul(&a.inner.borrow().data, &b.inner.borrow().data);
    Tensor {
        inner: Rc::new(RefCell::new(TensorInner {
            data: result_data,
            grad: None,
            _children: vec![a.clone(), b.clone()],
            _op: Some(Box::new(MatMul)),
        })),
    }
}

pub fn sub<B: Backend + Default>(a: &Tensor<B>, b: &Tensor<B>) -> Tensor<B> {
    let backend = B::default();
    let result_data = backend.sub(&a.inner.borrow().data, &b.inner.borrow().data);
    Tensor {
        inner: Rc::new(RefCell::new(TensorInner {
            data: result_data,
            grad: None,
            _children: vec![a.clone(), b.clone()],
            _op: Some(Box::new(Sub)),
        })),
    }
}

pub fn pow<B: Backend + Default>(a: &Tensor<B>, power: f32) -> Tensor<B> {
    let backend = B::default();
    let power_tensor: Tensor<B> = Tensor::new(&[power], &[1]);
    let result_data = backend.pow(&a.inner.borrow().data, &power_tensor.inner.borrow().data);
    Tensor {
        inner: Rc::new(RefCell::new(TensorInner {
            data: result_data,
            grad: None,
            _children: vec![a.clone()],
            _op: Some(Box::new(Pow { power })),
        })),
    }
}

pub fn mean<B: Backend + Default>(a: &Tensor<B>) -> Tensor<B> {
    let backend = B::default();
    let result_data = backend.mean(&a.inner.borrow().data);
    Tensor {
        inner: Rc::new(RefCell::new(TensorInner {
            data: result_data,
            grad: None,
            _children: vec![a.clone()],
            _op: Some(Box::new(Mean)),
        })),
    }
}

// --- Add the operator overloads ---
impl<B: Backend + Default> std::ops::Sub for &Tensor<B> {
    type Output = Tensor<B>;
    fn sub(self, rhs: Self) -> Self::Output {
        sub(self, rhs)
    }
}
