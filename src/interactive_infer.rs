use crate::dataset::is_prime;
use crate::matrix::Matrix;
use crate::model::Model;
use std::io::{self, Write};

/// Convert a number to its 10-bit binary vector
fn number_to_input(n: u16) -> Vec<u8> {
    (0..10).rev().map(|i| ((n >> i) & 1) as u8).collect()
}

/// Run interactive forward pass
pub fn run_inference(model: &mut Model) {
    loop {
        print!("🔢 Enter a number (0-1023) or 'q' to quit: ");
        io::stdout().flush().unwrap();

        let mut input_line = String::new();
        io::stdin().read_line(&mut input_line).unwrap();
        let input_line = input_line.trim();

        if input_line.eq_ignore_ascii_case("q") {
            break;
        }

        let number: u16 = match input_line.parse() {
            Ok(n) if n <= 1023 => n,
            _ => {
                println!("❌ Please enter a valid number (0–1023).");
                continue;
            }
        };

        let binary = number_to_input(number);
        println!("📥 Binary Input: {:?}", binary);

        let mut x = Matrix::from_vec(vec![binary.iter().map(|&b| b as f32).collect()]);

        for (i, layer) in model.layers_mut().enumerate() {
            x = layer.forward(&x);

            let label = match i {
                0 => "Input Projection",
                1 => "Activation 1",
                2 => "Hidden Layer 1",
                3 => "Activation 2",
                4 => "Hidden Layer 2",
                5 => "Activation 3",
                6 => "Hidden Layer 3",
                7 => "Activation 4",
                8 => "Output Projection",
                9 => "Output Activation",
                _ => "Unknown",
            };

            let row = x.row(0);
            let values = row
                .iter()
                .map(|v| format!("{:>6.2}", v))
                .collect::<Vec<_>>();

            println!("🧠 {:<18}: {}", label, values.join(" "));
        }

        let pred = x.get(0, 0);
        let actual = is_prime(number);

        println!(
            "🎯 Prediction: {:.4} → {}",
            pred,
            if pred >= 0.5 { "Prime?" } else { "Not Prime?" }
        );
        println!(
            "✅ Actual: {}\n",
            if actual { "Prime" } else { "Not Prime" }
        );
    }
}
