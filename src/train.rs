use crate::dataset::generate_dataset;
use crate::loss::mean_squared_error;
use crate::matrix::Matrix;
use crate::model::Model;

pub fn train_example() {
    let data = generate_dataset();

    // Convert to Matrix format
    let inputs = data
        .iter()
        .map(|(x, _)| x.iter().map(|&b| b as f32).collect::<Vec<f32>>())
        .map(|row| row)
        .collect::<Vec<Vec<f32>>>();

    let targets = data
        .iter()
        .map(|(_, y)| vec![*y as f32])
        .collect::<Vec<Vec<f32>>>();

    let input_matrix = Matrix::from_vec(inputs);
    let target_matrix = Matrix::from_vec(targets);

    // Build model
    let mut model = Model::new();
    model.add_layer(crate::layer::DenseLayer::new(10, 16));
    model.add_layer(crate::layer::ActivationLayer::new(
        crate::activation::sigmoid,
    ));
    model.add_layer(crate::layer::DenseLayer::new(16, 1));
    model.add_layer(crate::layer::ActivationLayer::new(
        crate::activation::sigmoid,
    ));

    // Forward pass
    let output = model.forward(&input_matrix);
    let loss = mean_squared_error(&output, &target_matrix);

    println!("Initial loss: {:.6}", loss);
}
