use crate::layer::Layer;
use crate::matrix::Matrix;

pub struct Model {
    layers: Vec<Box<dyn Layer>>,
}

impl Model {
    pub fn new() -> Self {
        Model { layers: Vec::new() }
    }

    pub fn add_layer<L: Layer + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer));
    }

    pub fn forward(&self, input: &Matrix) -> Matrix {
        self.layers
            .iter()
            .fold(input.clone(), |acc, layer| layer.forward(&acc))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::sigmoid;
    use crate::layer::{ActivationLayer, DenseLayer};

    #[test]
    fn test_model_forward_pass() {
        let mut model = Model::new();
        let mut dense = DenseLayer::new(2, 2);

        dense.set_weights(Matrix::from_vec(vec![vec![1.0, 0.0], vec![0.0, 1.0]]));
        dense.set_biases(Matrix::from_vec(vec![vec![0.0, 0.0]]));

        model.add_layer(dense);
        model.add_layer(ActivationLayer::new(sigmoid));

        let input = Matrix::from_vec(vec![vec![0.0, 2.0]]);

        let output = model.forward(&input);
        assert!((output.get(0, 0) - 0.5).abs() < 1e-5);
        assert!(output.get(0, 1) > 0.85); // sigmoid(2)
    }
}
