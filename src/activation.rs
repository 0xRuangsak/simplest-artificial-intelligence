/// Sigmoid activation function
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        let approx = sigmoid(0.0);
        assert!((approx - 0.5).abs() < 1e-5);
    }
}
