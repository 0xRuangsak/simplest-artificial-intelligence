mod activation;
mod dataset;
mod interactive_infer;
mod layer;
mod loss;
mod matrix;
mod model;
mod train;

use crate::activation::sigmoid;
use crate::layer::{ActivationLayer, DenseLayer};
use crate::model::Model;

fn main() {
    let mut model = Model::new();
    // Build your model and load trained weights if any
    // For now, rebuild architecture with random weights:

    use crate::activation::sigmoid;
    use crate::layer::{ActivationLayer, DenseLayer};

    model.add_layer(DenseLayer::new(10, 16));
    model.add_layer(ActivationLayer::new(sigmoid));
    model.add_layer(DenseLayer::new(16, 16));
    model.add_layer(ActivationLayer::new(sigmoid));
    model.add_layer(DenseLayer::new(16, 16));
    model.add_layer(ActivationLayer::new(sigmoid));
    model.add_layer(DenseLayer::new(16, 16));
    model.add_layer(ActivationLayer::new(sigmoid));
    model.add_layer(DenseLayer::new(16, 1));
    model.add_layer(ActivationLayer::new(sigmoid));

    interactive_infer::run_inference(&mut model);
}
