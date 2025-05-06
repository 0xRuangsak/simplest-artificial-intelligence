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
}
