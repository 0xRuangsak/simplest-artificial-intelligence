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
}
