use crate::activation::sigmoid;
use crate::dataset::generate_dataset;
use crate::layer::{ActivationLayer, DenseLayer};
use crate::loss::mean_squared_error;
use crate::matrix::Matrix;
use crate::model::Model;
use crate::train::train_step;
use std::io::{self, Write};

pub fn run_ui() {
    loop {
        println!("\n🤖 Simplest AI Interface");
        println!("1. Train the model");
        println!("2. Use the model");
        println!("3. Evaluate performance");
        println!("4. Quit");
        print!("Choose an option: ");
        io::stdout().flush().unwrap();

        let mut choice = String::new();
        io::stdin().read_line(&mut choice).unwrap();
        match choice.trim() {
            "1" => train_menu(),
            "2" => infer_menu(),
            "3" => evaluate_menu(),
            "4" => break,
            _ => println!("Invalid option. Try again."),
        }
    }
}

fn build_model() -> Model {
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
    model
}

fn train_menu() {
    let mut input = String::new();
    print!("\n📚 Enter number of training samples (1-1024): ");
    io::stdout().flush().unwrap();
    io::stdin().read_line(&mut input).unwrap();
    let count: usize = input.trim().parse().unwrap_or(0);
    if count == 0 || count > 1024 {
        println!("⚠️ Invalid sample count.");
        return;
    }

    let data = generate_dataset();
    let mut model = build_model();
    let learning_rate = 0.1;

    let mut total_loss = 0.0;
    for (x, y) in data.iter().take(count) {
        let input = Matrix::from_vec(vec![x.iter().map(|&b| b as f32).collect()]);
        let target = Matrix::from_vec(vec![vec![*y as f32]]);
        total_loss += train_step(&mut model, &input, &target, learning_rate);
    }

    println!(
        "✅ Training complete. Avg loss: {:.6}",
        total_loss / count as f32
    );
    // Optionally: save model to file or global state
}

fn infer_menu() {
    println!("⚙️ Inference logic not yet implemented here. Use existing run_inference().");
    // You can later move your existing inference code here
}

fn evaluate_menu() {
    println!("📊 Evaluation logic not yet implemented.");
    // Add accuracy, confusion matrix here later
}
