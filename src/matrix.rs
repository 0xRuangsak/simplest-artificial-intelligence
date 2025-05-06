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

    pub fn row(&self, index: usize) -> &[f32] {
        &self.data[index]
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
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

    pub fn dot(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "Dimension mismatch for dot product");

        let data = (0..self.rows)
            .map(|i| {
                (0..other.cols)
                    .map(|j| {
                        (0..self.cols)
                            .map(|k| self.get(i, k) * other.get(k, j))
                            .sum()
                    })
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>();

        Matrix::from_vec(data)
    }

    pub fn map<F>(&self, f: F) -> Matrix
    where
        F: Fn(f32) -> f32,
    {
        let data = self
            .data
            .iter()
            .map(|row| row.iter().map(|&val| f(val)).collect())
            .collect();

        Matrix::from_vec(data)
    }

    pub fn map_rows<F>(&self, f: F) -> Vec<Vec<f32>>
    where
        F: Fn(&[f32]) -> Vec<f32>,
    {
        self.data.iter().map(|row| f(row)).collect()
    }

    pub fn print(&self, label: &str) {
        println!("{} ({}x{}):", label, self.rows, self.cols);
        for (i, row) in self.data.iter().enumerate() {
            let formatted: Vec<String> = row.iter().map(|v| format!("{:>7.4}", v)).collect();
            println!("Row {:>3}: {}", i, formatted.join(" "));
        }
    }

    pub fn transpose(&self) -> Matrix {
        let data = (0..self.cols)
            .map(|j| (0..self.rows).map(|i| self.get(i, j)).collect())
            .collect();
        Matrix::from_vec(data)
    }

    pub fn sum_rows(&self) -> Matrix {
        let sums = (0..self.cols)
            .map(|j| (0..self.rows).map(|i| self.get(i, j)).sum())
            .collect();
        Matrix::from_vec(vec![sums])
    }

    pub fn data(&self) -> &Vec<Vec<f32>> {
        &self.data
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

    #[test]
    fn test_matrix_dot() {
        let a = Matrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        let b = Matrix::from_vec(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);

        // Expected:
        // [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
        // [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
        let result = a.dot(&b);

        assert_eq!(result.get(0, 0), 19.0);
        assert_eq!(result.get(0, 1), 22.0);
        assert_eq!(result.get(1, 0), 43.0);
        assert_eq!(result.get(1, 1), 50.0);
    }

    #[test]
    fn test_matrix_map() {
        let m = Matrix::from_vec(vec![vec![-1.0, 0.0], vec![1.0, 2.0]]);

        let result = m.map(|x| x * x); // square all elements

        assert_eq!(result.get(0, 0), 1.0);
        assert_eq!(result.get(0, 1), 0.0);
        assert_eq!(result.get(1, 0), 1.0);
        assert_eq!(result.get(1, 1), 4.0);
    }
}
