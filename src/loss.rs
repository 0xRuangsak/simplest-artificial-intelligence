use crate::matrix::Matrix;

pub fn mean_squared_error(predicted: &Matrix, target: &Matrix) -> f32 {
    assert_eq!(predicted.rows(), target.rows());
    assert_eq!(predicted.cols(), target.cols());

    let total: f32 = (0..predicted.rows())
        .map(|i| {
            (0..predicted.cols())
                .map(|j| {
                    let diff = predicted.get(i, j) - target.get(i, j);
                    diff * diff
                })
                .sum::<f32>()
        })
        .sum();

    total / (predicted.rows() * predicted.cols()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Matrix;

    #[test]
    fn test_mse() {
        let predicted = Matrix::from_vec(vec![vec![0.1], vec![0.9]]);
        let target = Matrix::from_vec(vec![vec![0.0], vec![1.0]]);
        let loss = mean_squared_error(&predicted, &target);
        assert!((loss - 0.01).abs() < 1e-5);
    }
}
