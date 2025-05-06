use crate::layer::{DenseLayer, LayerEnum};
use crate::matrix::Matrix;

pub struct Model {
    layers: Vec<LayerEnum>,
}

impl Model {
    pub fn new() -> Self {
        Model { layers: Vec::new() }
    }

    pub fn add_layer(&mut self, layer: LayerEnum) {
        self.layers.push(layer);
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

    pub fn print_weights(&self) {
        for (i, layer) in self.layers.iter().enumerate() {
            println!("🔧 Layer {} Weights:", i);
            for (i, layer) in self.layers.iter().enumerate() {
                layer.print_weights(i);
            }
        }
    }

    pub fn layers(&self) -> &[LayerEnum] {
        &self.layers
    }

    pub fn layers_mut(&mut self) -> std::slice::IterMut<'_, LayerEnum> {
        self.layers.iter_mut()
    }

    pub fn set_weight(&mut self, layer_index: usize, row: usize, col: usize, value: f32) {
        if let Some(LayerEnum::Dense(layer)) = self.layers.get_mut(layer_index) {
            layer.weights.set(row, col, value);
        } else {
            println!("⚠️ Layer {} is not a Dense layer.", layer_index);
        }
    }

    pub fn randomize_all_weights(&mut self) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for layer in &mut self.layers {
            if let LayerEnum::Dense(d) = layer {
                for r in 0..d.weights.rows() {
                    for c in 0..d.weights.cols() {
                        d.weights.set(r, c, rng.gen_range(-1.0..1.0));
                    }
                }
            }
        }
        println!("🔀 All weights randomized.");
    }

    pub fn zero_all_weights(&mut self) {
        for layer in &mut self.layers {
            if let LayerEnum::Dense(d) = layer {
                for r in 0..d.weights.rows() {
                    for c in 0..d.weights.cols() {
                        d.weights.set(r, c, 0.0);
                    }
                }
            }
        }
        println!("🧹 All weights set to zero.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::sigmoid;
    use crate::layer::{ActivationLayer, LayerEnum};

    #[test]
    fn test_model_forward_pass() {
        let mut model = Model::new();
        let mut dense = DenseLayer::new(2, 2);

        dense.set_weights(Matrix::from_vec(vec![vec![1.0, 0.0], vec![0.0, 1.0]]));
        dense.set_biases(Matrix::from_vec(vec![vec![0.0, 0.0]]));

        model.add_layer(LayerEnum::Dense(dense));
        model.add_layer(LayerEnum::Activation(ActivationLayer::new(sigmoid)));

        let input = Matrix::from_vec(vec![vec![0.0, 2.0]]);
        let output = model.forward(&input);

        assert!((output.get(0, 0) - 0.5).abs() < 1e-5);
        assert!(output.get(0, 1) > 0.85); // sigmoid(2)
    }
}
