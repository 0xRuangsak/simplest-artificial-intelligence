use crate::matrix::Matrix;
use rand::Rng;

pub struct DenseLayer {
    input_size: usize,
    output_size: usize,
    weights: Matrix,
    biases: Matrix,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Matrix::from_vec(
            (0..input_size)
                .map(|_| (0..output_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
                .collect(),
        );

        let biases = Matrix::new(1, output_size); // all zeros

        DenseLayer {
            input_size,
            output_size,
            weights,
            biases,
        }
    }

    pub fn forward(&self, input: &Matrix) -> Matrix {
        let dot = input.dot(&self.weights);

        // Add biases to each row (broadcasting)
        let output = dot.map_rows(|row| {
            row.iter()
                .zip(self.biases.row(0).iter())
                .map(|(x, b)| x + b)
                .collect::<Vec<f32>>()
        });

        Matrix::from_vec(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
