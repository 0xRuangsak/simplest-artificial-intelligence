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

    pub fn forward(&mut self, input: &Matrix) -> Matrix {
        self.layers
            .iter_mut()
            .fold(input.clone(), |acc, layer| layer.forward(&acc))
    }

    pub fn forward_verbose(&mut self, input: &Matrix) -> Matrix {
        let mut x = input.clone();
        x.print("📥 Input");

        for (i, layer) in self.layers.iter_mut().enumerate() {
            x = layer.forward(&x);
            x.print(&format!("Layer {}", i));
        }

        x
    }

    pub fn layers(&self) -> &[Box<dyn Layer>] {
        &self.layers
    }

    pub fn layers_mut(&mut self) -> std::slice::IterMut<'_, Box<dyn Layer>> {
        self.layers.iter_mut()
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
