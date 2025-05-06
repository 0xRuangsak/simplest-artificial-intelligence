use crate::activation::sigmoid;
use crate::matrix::Matrix;

pub trait Layer: LayerClone {
    fn forward(&mut self, input: &Matrix) -> Matrix;
    fn backward(&mut self, input: &Matrix, grad_output: &Matrix) -> Matrix;
    fn update(&mut self, learning_rate: f32);
}

pub trait LayerClone {
    fn clone_box(&self) -> Box<dyn Layer>;
}

impl<T: 'static + Layer + Clone> LayerClone for T {
    fn clone_box(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Layer> {
    fn clone(&self) -> Box<dyn Layer> {
        self.clone_box()
    }
}

#[derive(Debug, Clone)]
pub struct DenseLayer {
    input_size: usize,
    output_size: usize,
    weights: Matrix,
    biases: Matrix,
    last_input: Option<Matrix>,
    grad_weights: Option<Matrix>,
    grad_biases: Option<Matrix>,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let weights = Matrix::from_vec(
            (0..input_size)
                .map(|_| (0..output_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
                .collect(),
        );

        let biases = Matrix::new(1, output_size);
        DenseLayer {
            input_size,
            output_size,
            weights,
            biases,
            last_input: None,
            grad_weights: None,
            grad_biases: None,
        }
    }

    pub fn set_weights(&mut self, weights: Matrix) {
        self.weights = weights;
    }

    pub fn set_biases(&mut self, biases: Matrix) {
        self.biases = biases;
    }
}

impl Layer for DenseLayer {
    fn forward(&mut self, input: &Matrix) -> Matrix {
        let dot = input.dot(&self.weights);
        let output = dot.map_rows(|row| {
            row.iter()
                .zip(self.biases.row(0).iter())
                .map(|(x, b)| x + b)
                .collect()
        });
        Matrix::from_vec(output)
    }

    fn backward(&mut self, input: &Matrix, grad_output: &Matrix) -> Matrix {
        self.last_input = Some(input.clone());

        let grad_w = input.transpose().dot(grad_output);
        let grad_b = grad_output.sum_rows();

        self.grad_weights = Some(grad_w);
        self.grad_biases = Some(grad_b);

        grad_output.dot(&self.weights.transpose())
    }

    fn update(&mut self, learning_rate: f32) {
        if let Some(gw) = &self.grad_weights {
            self.weights = self.weights.add(&gw.map(|v| -learning_rate * v));
        }
        if let Some(gb) = &self.grad_biases {
            self.biases = self.biases.add(&gb.map(|v| -learning_rate * v));
        }
    }
}

#[derive(Clone)]
pub struct ActivationLayer {
    activation_fn: fn(f32) -> f32,
    last_output: Option<Matrix>,
}

impl ActivationLayer {
    pub fn new(activation_fn: fn(f32) -> f32) -> Self {
        Self {
            activation_fn,
            last_output: None,
        }
    }

    fn activation_derivative(&self, y: f32) -> f32 {
        y * (1.0 - y) // sigmoid derivative
    }
}

impl Layer for ActivationLayer {
    fn forward(&mut self, input: &Matrix) -> Matrix {
        let output = input.map(self.activation_fn);
        self.last_output = Some(output.clone());
        output
    }

    fn backward(&mut self, _input: &Matrix, grad_output: &Matrix) -> Matrix {
        let cached = self.last_output.as_ref().expect("Missing cached output");
        let result = cached
            .data()
            .iter()
            .zip(grad_output.data().iter())
            .map(|(a, b)| {
                a.iter()
                    .zip(b.iter())
                    .map(|(&y, &dy)| dy * self.activation_derivative(y))
                    .collect()
            })
            .collect();
        Matrix::from_vec(result)
    }

    fn update(&mut self, _learning_rate: f32) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::sigmoid;

    #[test]
    fn test_dense_layer_creation() {
        let layer = DenseLayer::new(10, 16);
        assert_eq!(layer.weights.rows(), 10);
        assert_eq!(layer.weights.cols(), 16);
        assert_eq!(layer.biases.rows(), 1);
        assert_eq!(layer.biases.cols(), 16);
    }

    #[test]
    fn test_forward_no_activation() {
        let input = Matrix::from_vec(vec![vec![1.0, 2.0]]);
        let weights = Matrix::from_vec(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        let biases = Matrix::from_vec(vec![vec![0.5, -0.5]]);

        let mut layer = DenseLayer::new(2, 2);
        layer.weights = weights;
        layer.biases = biases;

        let output = layer.forward(&input);
        assert_eq!(output.get(0, 0), 1.0 + 0.5);
        assert_eq!(output.get(0, 1), 2.0 - 0.5);
    }

    #[test]
    fn test_layer_trait_forward() {
        let input = Matrix::from_vec(vec![vec![1.0, 2.0]]);
        let weights = Matrix::from_vec(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        let biases = Matrix::from_vec(vec![vec![0.5, -0.5]]);

        let mut layer = DenseLayer::new(2, 2);
        layer.weights = weights;
        layer.biases = biases;

        let output = layer.forward(&input); // using &mut self now
        assert_eq!(output.get(0, 0), 1.5);
        assert_eq!(output.get(0, 1), 1.5);
    }

    #[test]
    fn test_activation_layer_sigmoid() {
        let input = Matrix::from_vec(vec![vec![0.0, 2.0], vec![-2.0, 1.0]]);
        let mut layer = ActivationLayer::new(sigmoid);
        let output = layer.forward(&input);

        assert!((output.get(0, 0) - 0.5).abs() < 1e-5);
        assert!(output.get(0, 1) > 0.85);
        assert!(output.get(1, 0) < 0.2);
    }

    #[test]
    fn test_dense_layer_backward_shapes() {
        let mut layer = DenseLayer::new(2, 3);
        let input = Matrix::from_vec(vec![vec![1.0, 2.0]]);
        let grad_output = Matrix::from_vec(vec![vec![0.1, 0.2, 0.3]]);
        let grad_input = layer.backward(&input, &grad_output);

        assert_eq!(grad_input.rows(), 1);
        assert_eq!(grad_input.cols(), 2);

        let gw = layer.grad_weights.as_ref().unwrap();
        let gb = layer.grad_biases.as_ref().unwrap();

        assert_eq!(gw.rows(), 2);
        assert_eq!(gw.cols(), 3);
        assert_eq!(gb.rows(), 1);
        assert_eq!(gb.cols(), 3);
    }
}
