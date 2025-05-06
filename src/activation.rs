fn sigmoid_derivative(y: f32) -> f32 {
    y * (1.0 - y)
}

/// Sigmoid activation function
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// ReLU activation function
pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

pub fn leaky_relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.01 * x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        let approx = sigmoid(0.0);
        assert!((approx - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_relu() {
        assert_eq!(relu(0.0), 0.0);
        assert_eq!(relu(1.5), 1.5);
        assert_eq!(relu(-2.0), 0.0);
    }
}
