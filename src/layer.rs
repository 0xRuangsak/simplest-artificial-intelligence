use crate::matrix::Matrix;

#[derive(Clone)]
pub enum LayerEnum {
    Dense(DenseLayer),
    Activation(ActivationLayer),
}

impl LayerEnum {
    pub fn forward(&mut self, input: &Matrix) -> Matrix {
        match self {
            LayerEnum::Dense(layer) => layer.forward(input),
            LayerEnum::Activation(layer) => layer.forward(input),
        }
    }

    pub fn backward(&mut self, input: &Matrix, grad_output: &Matrix) -> Matrix {
        match self {
            LayerEnum::Dense(layer) => layer.backward(input, grad_output),
            LayerEnum::Activation(layer) => layer.backward(input, grad_output),
        }
    }

    pub fn update(&mut self, learning_rate: f32) {
        match self {
            LayerEnum::Dense(layer) => layer.update(learning_rate),
            LayerEnum::Activation(layer) => layer.update(learning_rate),
        }
    }

    pub fn print_weights(&self, index: usize) {
        match self {
            LayerEnum::Dense(layer) => {
                println!("📊 Dense Layer {} Weights:", index);
                layer.weights.print("Weights");
                layer.biases.print("Biases");
            }
            LayerEnum::Activation(_) => {
                println!("⚙️ Activation Layer {} (no weights)", index);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub weights: Matrix,
    pub biases: Matrix,
    pub last_input: Option<Matrix>,
    pub grad_weights: Option<Matrix>,
    pub grad_biases: Option<Matrix>,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let weights = Matrix::from_vec(
            (0..input_size)
                .map(|_| (0..output_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
                .collect(),
        );

        let biases = Matrix::new(1, output_size);
        Self {
            weights,
            biases,
            last_input: None,
            grad_weights: None,
            grad_biases: None,
        }
    }

    pub fn forward(&mut self, input: &Matrix) -> Matrix {
        let dot = input.dot(&self.weights);
        let output = dot.map_rows(|row| {
            row.iter()
                .zip(self.biases.row(0).iter())
                .map(|(x, b)| x + b)
                .collect()
        });
        Matrix::from_vec(output)
    }

    pub fn backward(&mut self, input: &Matrix, grad_output: &Matrix) -> Matrix {
        self.last_input = Some(input.clone());
        let grad_w = input.transpose().dot(grad_output);
        let grad_b = grad_output.sum_rows();
        self.grad_weights = Some(grad_w);
        self.grad_biases = Some(grad_b);
        grad_output.dot(&self.weights.transpose())
    }

    pub fn update(&mut self, learning_rate: f32) {
        if let Some(gw) = &self.grad_weights {
            self.weights = self.weights.add(&gw.map(|v| -learning_rate * v));
        }
        if let Some(gb) = &self.grad_biases {
            self.biases = self.biases.add(&gb.map(|v| -learning_rate * v));
        }
    }
}

#[derive(Clone)]
pub struct ActivationLayer {
    activation_fn: fn(f32) -> f32,
    last_output: Option<Matrix>,
}

impl ActivationLayer {
    pub fn new(activation_fn: fn(f32) -> f32) -> Self {
        Self {
            activation_fn,
            last_output: None,
        }
    }

    fn activation_derivative(&self, y: f32) -> f32 {
        y * (1.0 - y)
    }

    pub fn forward(&mut self, input: &Matrix) -> Matrix {
        let output = input.map(self.activation_fn);
        self.last_output = Some(output.clone());
        output
    }

    pub fn backward(&mut self, _input: &Matrix, grad_output: &Matrix) -> Matrix {
        let cached = self.last_output.as_ref().expect("Missing cached output");
        let result = cached
            .data()
            .iter()
            .zip(grad_output.data().iter())
            .map(|(a, b)| {
                a.iter()
                    .zip(b.iter())
                    .map(|(&y, &dy)| dy * self.activation_derivative(y))
                    .collect()
            })
            .collect();
        Matrix::from_vec(result)
    }

    pub fn update(&mut self, _learning_rate: f32) {}
}
