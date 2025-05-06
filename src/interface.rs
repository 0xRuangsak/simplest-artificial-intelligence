use crate::dataset::generate_dataset;
use crate::loss::mean_squared_error;
use crate::matrix::Matrix;
use crate::train::{debug_forward_sample, train_step};
use std::io::{self, Write};

use crate::activation::sigmoid;
use crate::layer::{ActivationLayer, DenseLayer, LayerEnum};
use crate::model::Model;

pub fn run_ui() {
    let mut model = build_model();

    loop {
        println!("\n🤖 Simplest AI Interface");
        println!("1. Train the model");
        println!("2. Use the model");
        println!("3. Evaluate performance");
        println!("4. View model weights");
        println!("5. Edit model weights");
        println!("6. Quit");
        print!("Choose an option: ");
        io::stdout().flush().unwrap();

        let mut choice = String::new();
        io::stdin().read_line(&mut choice).unwrap();
        match choice.trim() {
            "1" => train_menu(&mut model),
            "2" => infer_menu(&mut model),
            "3" => evaluate_menu(&mut model),
            "4" => model.print_weights(),
            "5" => edit_weights_menu(&mut model),
            "6" => break,
            _ => println!("Invalid option. Try again."),
        }
    }
}

fn build_model() -> Model {
    let mut model = Model::new();
    model.add_layer(LayerEnum::Dense(DenseLayer::new(10, 16)));
    model.add_layer(LayerEnum::Activation(ActivationLayer::new(sigmoid)));
    model.add_layer(LayerEnum::Dense(DenseLayer::new(16, 16)));
    model.add_layer(LayerEnum::Activation(ActivationLayer::new(sigmoid)));
    model.add_layer(LayerEnum::Dense(DenseLayer::new(16, 16)));
    model.add_layer(LayerEnum::Activation(ActivationLayer::new(sigmoid)));
    model.add_layer(LayerEnum::Dense(DenseLayer::new(16, 16)));
    model.add_layer(LayerEnum::Activation(ActivationLayer::new(sigmoid)));
    model.add_layer(LayerEnum::Dense(DenseLayer::new(16, 1)));
    model.add_layer(LayerEnum::Activation(ActivationLayer::new(sigmoid)));
    model
}

fn train_menu(model: &mut Model) {
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
    let learning_rate = 0.1;

    let mut total_loss = 0.0;
    for (x, y) in data.iter().take(count) {
        let input = Matrix::from_vec(vec![x.iter().map(|&b| b as f32).collect()]);
        let target = Matrix::from_vec(vec![vec![*y as f32]]);
        total_loss += train_step(model, &input, &target, learning_rate);
    }

    println!(
        "✅ Training complete. Avg loss: {:.6}",
        total_loss / count as f32
    );
}

fn infer_menu(model: &mut Model) {
    loop {
        print!("\n🔢 Enter a number (0-1023) or 'q' to quit: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input.eq_ignore_ascii_case("q") {
            break;
        }

        if let Ok(num) = input.parse::<usize>() {
            if num <= 1023 {
                let binary_input = (0..10)
                    .rev()
                    .map(|i| ((num >> i) & 1) as u8)
                    .collect::<Vec<_>>();
                let is_prime = crate::dataset::is_prime(num as u16);
                let label = if is_prime { 1 } else { 0 };
                debug_forward_sample(model, &binary_input, label, num);
            } else {
                println!("⚠️ Number out of range (0–1023).");
            }
        } else {
            println!("⚠️ Invalid input.");
        }
    }
}

fn evaluate_menu(model: &mut Model) {
    let data = generate_dataset();
    let mut correct = 0;
    let mut true_pos = vec![];
    let mut false_pos = vec![];
    let mut true_neg = vec![];
    let mut false_neg = vec![];

    for (i, (x, y)) in data.iter().enumerate() {
        let input = Matrix::from_vec(vec![x.iter().map(|&b| b as f32).collect()]);
        let output = model.forward(&input);
        let prediction = output.get(0, 0);

        if prediction >= 0.5 && *y == 1 {
            correct += 1;
            true_pos.push(i);
        } else if prediction < 0.5 && *y == 0 {
            correct += 1;
            true_neg.push(i);
        } else if prediction >= 0.5 && *y == 0 {
            false_pos.push(i);
        } else {
            false_neg.push(i);
        }
    }

    let accuracy = 100.0 * correct as f32 / data.len() as f32;
    println!("\n📊 Evaluation Results:");
    println!("✅ Accuracy: {:.2}%", accuracy);
    println!("✔️ True Positives: {:?}", true_pos);
    println!("❌ False Positives: {:?}", false_pos);
    println!("✔️ True Negatives: {:?}", true_neg);
    println!("❌ False Negatives: {:?}", false_neg);
}

fn edit_weights_menu(model: &mut Model) {
    loop {
        println!("\n🛠️ Edit Weights Menu");
        println!("1. Manually edit specific weight");
        println!("2. Randomize all weights");
        println!("3. Set all weights to zero");
        println!("4. Back to main menu");
        print!("Choose an option: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        match input.trim() {
            "1" => {
                println!("🔧 Manually editing a weight.");
                print!("Enter Dense layer index: ");
                io::stdout().flush().unwrap();
                let mut idx = String::new();
                io::stdin().read_line(&mut idx).unwrap();
                let idx: usize = idx.trim().parse().unwrap_or(0);

                print!("Enter weight row index: ");
                io::stdout().flush().unwrap();
                let mut row = String::new();
                io::stdin().read_line(&mut row).unwrap();
                let row: usize = row.trim().parse().unwrap_or(0);

                print!("Enter weight column index: ");
                io::stdout().flush().unwrap();
                let mut col = String::new();
                io::stdin().read_line(&mut col).unwrap();
                let col: usize = col.trim().parse().unwrap_or(0);

                print!("Enter new value: ");
                io::stdout().flush().unwrap();
                let mut val = String::new();
                io::stdin().read_line(&mut val).unwrap();
                let val: f32 = val.trim().parse().unwrap_or(0.0);

                model.set_weight(idx, row, col, val);
                println!("✅ Weight updated.");
            }
            "2" => {
                model.randomize_all_weights();
                println!("🎲 All weights randomized.");
            }
            "3" => {
                model.zero_all_weights();
                println!("🧼 All weights set to zero.");
            }
            "4" => break,
            _ => println!("Invalid option. Try again."),
        }
    }
}
