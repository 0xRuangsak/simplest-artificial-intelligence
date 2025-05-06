use crate::activation::sigmoid;
use crate::dataset::generate_dataset;
use crate::layer::{ActivationLayer, DenseLayer};
use crate::loss::mean_squared_error;
use crate::matrix::Matrix;
use crate::model::Model;

pub fn train_example() {
    let data = generate_dataset();

    let mut model = Model::new();
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
    let learning_rate = 0.1;
    let epochs = 20;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for (x, y) in &data {
            let input = Matrix::from_vec(vec![x.iter().map(|&b| b as f32).collect()]);
            let target = Matrix::from_vec(vec![vec![*y as f32]]);
            total_loss += train_step(&mut model, &input, &target, learning_rate);
        }

        let mut correct = 0;

        for (x, y) in &data {
            let input = Matrix::from_vec(vec![x.iter().map(|&b| b as f32).collect()]);
            let output = model.forward(&input);
            let prediction = output.get(0, 0);

            if (prediction >= 0.5 && *y == 1) || (prediction < 0.5 && *y == 0) {
                correct += 1;
            }
        }

        let accuracy = 100.0 * correct as f32 / data.len() as f32;
        println!(
            "📚 Epoch {:>3} — Loss: {:.6} — Accuracy: {:>5.2}%",
            epoch + 1,
            total_loss / data.len() as f32,
            accuracy
        );
    }

    println!("✅ Training complete.");
}

pub fn train_step(model: &mut Model, input: &Matrix, target: &Matrix, learning_rate: f32) -> f32 {
    let output = model.forward(input);
    let loss = mean_squared_error(&output, target);

    let mut x = input.clone();
    let mut intermediates = vec![x.clone()];
    for layer in model.layers_mut() {
        x = layer.forward(&x);
        intermediates.push(x.clone());
    }

    let mut grad = output.add(&target.map(|v| -v)).map(|v| 2.0 * v);

    for (layer, input) in model
        .layers_mut()
        .rev()
        .zip(intermediates.iter().rev().skip(1))
    {
        grad = layer.backward(input, &grad);
    }

    for layer in model.layers_mut() {
        layer.update(learning_rate);
    }

    loss
}

pub fn debug_forward_sample(model: &mut Model, input: &[u8], label: u8, number: usize) {
    println!("==============================");
    println!("🔢 Number: {}", number);
    println!("📥 Binary Input: {:?}", input);

    let mut x = Matrix::from_vec(vec![input.iter().map(|&b| b as f32).collect()]);
    x.print("Input");

    for (i, layer) in model.layers_mut().enumerate() {
        x = layer.forward(&x);
        x.print(&format!("Layer {}", i));
    }

    let prediction = x.get(0, 0);
    println!("🎯 Target: {}", label);
    println!("🔮 Final Prediction: {:.4}", prediction);
}
