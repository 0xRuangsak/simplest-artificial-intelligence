pub fn u16_to_bin_vector(n: u16) -> Vec<u8> {
    (0..10).rev().map(|i| ((n >> i) & 1) as u8).collect()
}

pub fn is_prime(n: u16) -> bool {
    if n < 2 {
        return false;
    }
    for i in 2..=((n as f64).sqrt() as u16) {
        if n % i == 0 {
            return false;
        }
    }
    true
}

pub fn generate_dataset() -> Vec<(Vec<u8>, u8)> {
    (0..=1023)
        .map(|n| {
            let input = u16_to_bin_vector(n);
            let label = if is_prime(n) { 1 } else { 0 };
            (input, label)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u16_to_bin_vector() {
        assert_eq!(u16_to_bin_vector(0), vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(u16_to_bin_vector(1), vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
        assert_eq!(u16_to_bin_vector(1023), vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(u16_to_bin_vector(5), vec![0, 0, 0, 0, 0, 0, 0, 1, 0, 1]);
    }

    #[test]
    fn test_is_prime() {
        let primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29];
        let non_primes = [0, 1, 4, 6, 8, 9, 10, 12, 14, 15];

        for &p in primes.iter() {
            assert!(is_prime(p), "{} should be prime", p);
        }
        for &n in non_primes.iter() {
            assert!(!is_prime(n), "{} should not be prime", n);
        }
    }

    #[test]
    fn test_generate_dataset_size() {
        let data = generate_dataset();
        assert_eq!(data.len(), 1024);
    }
}
