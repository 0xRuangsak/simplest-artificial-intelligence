#[derive(Debug, Clone)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<Vec<f32>>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row][col]
    }

    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.data[row][col] = value;
    }

    pub fn from_vec(data: Vec<Vec<f32>>) -> Self {
        let rows = data.len();
        assert!(rows > 0, "Matrix must have at least one row");
        let cols = data[0].len();
        assert!(
            data.iter().all(|r| r.len() == cols),
            "All rows must have the same number of columns"
        );

        Matrix { rows, cols, data }
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows, "Row count mismatch");
        assert_eq!(self.cols, other.cols, "Column count mismatch");

        let data = (0..self.rows)
            .map(|i| {
                (0..self.cols)
                    .map(|j| self.get(i, j) + other.get(i, j))
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>();

        Matrix::from_vec(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_new_and_get_set() {
        let mut m = Matrix::new(2, 3);
        assert_eq!(m.get(0, 0), 0.0);
        m.set(0, 1, 1.5);
        assert_eq!(m.get(0, 1), 1.5);
        m.set(1, 2, -2.0);
        assert_eq!(m.get(1, 2), -2.0);
    }

    #[test]
    fn test_matrix_from_vec() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let m = Matrix::from_vec(data.clone());
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 2), 6.0);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
    }

    #[test]
    fn test_matrix_add() {
        let a = Matrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        let b = Matrix::from_vec(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);

        let result = a.add(&b);

        assert_eq!(result.get(0, 0), 6.0);
        assert_eq!(result.get(1, 1), 12.0);
    }
}
