# MiniTorch-rs: A Minimalist Deep Learning Framework in Rust

This project is an educational exercise to build a minimalist, PyTorch-inspired deep learning framework from scratch in pure, idiomatic Rust. The primary goal is not to compete with production frameworks like PyTorch or Burn, but to demystify the core architectural principles behind them.

By building this project, we aim to gain a deep, first-principles understanding of:
-   **Tensors:** The fundamental data structure for numerical computation.
-   **The Autograd Engine:** The "magic" of automatic differentiation via a dynamic computational graph.
-   **Neural Network Abstractions:** How layers, modules, and activation functions are structured.
-   **Training Loops:** The mechanics of optimizers, loss functions, and backpropagation.

The framework will be CPU-only, using the `ndarray` crate for its underlying mathematical operations.

---

## Project Philosophy and Scope

-   **Clarity over Performance:** The code will be written to be as clear and educational as possible. While performance is a consideration, we will always favor a simpler, more understandable implementation over a hyper-optimized but obscure one.
-   **Core Goal: CPU-Only MLP:** The primary, version 1.0 goal of this project is to build a CPU-only framework capable of training a simple Multi-Layer Perceptron (MLP).
-   **Focus on Autograd:** The heart of the project is the automatic differentiation engine. All other features serve this core goal.

---

## Definition of Done: The "XOR" Milestone

This project will be considered "complete" when it can successfully **train a simple Multi-Layer Perceptron to solve the classic XOR problem.**

This milestone is a perfect litmus test because achieving it requires every single core component of the framework to be working correctly:
1.  **Tensors:** To hold the data, weights, and gradients.
2.  **Operations:** At least `MatMul`, `Add`, and an activation function like `ReLU`.
3.  **Computational Graph:** To track the relationships between operations.
4.  **Autograd Engine:** The `backward()` pass must correctly calculate gradients using the chain rule.
5.  **Neural Network Layers:** A working `Linear` layer (`nn.Module`).
6.  **Loss Function:** A `MeanSquaredError` function.
7.  **Optimizer:** A functional `SGD` optimizer.
8.  **Training Loop:** The ability to put it all together to iteratively update the model's weights and reduce the loss.

When the `train_xor.rs` example can consistently converge and print a loss that approaches zero, the primary goal of this project will have been met.

### Phased Development Plan

The project will be built in the following phases:

-   **Phase 1: The Tensor and Backend Abstraction**
    -   [x] Create a `Tensor` struct wrapping an `ndarray`.
    -   [x] Define a `Backend` trait to separate logic from computation.
    -   [x] Implement basic operator overloading (e.g., `+`).

-   **Phase 2: The Computational Graph**
    -   [x] Augment `Tensor` to store its history (`_children` and `_op`).
    -   [x] Implement a `trait Op` for operations.
    -   [x] Implement the graph-building logic for `Add` and `Mul`.

-   **Phase 3: Backpropagation**
    -   [ ] Implement a topological sort for the computational graph.
    -   [ ] Implement the `Tensor.backward()` method.
    -   [ ] Implement the `backward()` pass for `Add` and `Mul`.

-   **Phase 4: Building a Neural Network**
    -   [ ] Implement `MatMul` and `ReLU` operations.
    -   [ ] Create a `trait Module` for neural network layers.
    -   [ ] Implement a `Linear` layer.
    -   [ ] Implement a `MeanSquaredError` loss function.

-   **Phase 5: Training**
    -   [ ] Implement an `SGD` optimizer.
    -   [ ] Create the final `train_xor.rs` example, putting all the pieces together to train a model.

This phased approach provides a clear roadmap and allows for incremental progress and testing at each stage.

### Stretch Goals (Post-V1.0)

Once the core framework is complete and proven with the XOR milestone, the following advanced features are exciting possibilities for future development:

-   **Expanded Operator Library:** Implementing more complex operations like 2D convolutions (`Conv2d`) to enable Convolutional Neural Networks (CNNs).
-   **GPU Backend:** Creating a `WgpuBackend` that implements the `Backend` trait, allowing the entire framework to run computations on the GPU.
-   **More Optimizers:** Adding other common optimizers like Adam.
-   **Serialization:** Adding the ability to save and load trained model weights.