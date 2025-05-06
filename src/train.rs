use crate::loss::mean_squared_error;
use crate::matrix::Matrix;
use crate::model::Model;

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
    println!("🌟 Target: {}", label);
    println!("🔮 Final Prediction: {:.4}", prediction);
}
