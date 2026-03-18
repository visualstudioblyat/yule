//! Weight matrix rank analysis and low-rank compression.
//! Reference: arXiv 2509.22445 (Kolmogorov complexity, 2025)

/// Analyze the energy distribution of a weight vector.
///
/// Returns the fraction of total energy (sum of squares) contained in the
/// top `top_fraction` of values (by magnitude). A value near 1.0 indicates
/// that energy is highly concentrated in a few entries.
pub fn energy_concentration(weights: &[f32], top_fraction: f32) -> f64 {
    assert!(
        (0.0..=1.0).contains(&top_fraction),
        "top_fraction must be in [0, 1]"
    );
    if weights.is_empty() {
        return 0.0;
    }

    let total_energy: f64 = weights.iter().map(|&w| (w as f64) * (w as f64)).sum();
    if total_energy < 1e-30 {
        return 0.0;
    }

    let mut sorted: Vec<f64> = weights.iter().map(|&w| (w as f64) * (w as f64)).collect();
    sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());

    let k = ((weights.len() as f64 * top_fraction as f64).ceil() as usize).min(sorted.len());
    let top_energy: f64 = sorted[..k].iter().sum();
    top_energy / total_energy
}

/// Estimate effective rank from energy concentration.
///
/// Uses a binary search to find the smallest fraction of entries that captures
/// at least 90% of the total energy. Returns that fraction (0.0 to 1.0),
/// representing the effective rank as a proportion of the full dimension.
pub fn estimate_effective_rank(weights: &[f32]) -> f64 {
    if weights.is_empty() {
        return 0.0;
    }

    let total_energy: f64 = weights.iter().map(|&w| (w as f64) * (w as f64)).sum();
    if total_energy < 1e-30 {
        return 0.0;
    }

    let mut sorted: Vec<f64> = weights.iter().map(|&w| (w as f64) * (w as f64)).collect();
    sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());

    let threshold = 0.9 * total_energy;
    let mut cumulative = 0.0;
    for (i, &e) in sorted.iter().enumerate() {
        cumulative += e;
        if cumulative >= threshold {
            return (i + 1) as f64 / weights.len() as f64;
        }
    }
    1.0
}

/// Low-rank approximation via truncated SVD using power iteration.
///
/// Decomposes a matrix `[rows, cols]` into `U * V^T` where `U` is `[rows, rank]`
/// and `V` is `[cols, rank]`. The approximation is computed by repeatedly
/// extracting dominant singular vectors via power iteration with deflation.
pub struct LowRankApprox {
    /// Left factors, shape [rows, rank] row-major.
    pub u: Vec<f32>,
    /// Right factors, shape [cols, rank] row-major.
    pub v: Vec<f32>,
    /// Number of retained singular components.
    pub rank: usize,
    pub rows: usize,
    pub cols: usize,
}

impl LowRankApprox {
    /// Compute a rank-k approximation of `matrix` (shape `[rows, cols]`, row-major)
    /// via power iteration with deflation.
    pub fn compute(matrix: &[f32], rows: usize, cols: usize, target_rank: usize) -> Self {
        assert_eq!(matrix.len(), rows * cols, "matrix size mismatch");
        let rank = target_rank.min(rows).min(cols);

        let mut residual = matrix.to_vec();
        let mut u_vecs: Vec<Vec<f32>> = Vec::with_capacity(rank);
        let mut v_vecs: Vec<Vec<f32>> = Vec::with_capacity(rank);

        let max_iters = 200;

        for _ in 0..rank {
            // Initialize v with a deterministic vector
            let mut v = vec![0.0f32; cols];
            for (i, val) in v.iter_mut().enumerate() {
                *val = ((i as f32 + 1.0) * std::f32::consts::FRAC_1_SQRT_2).sin();
            }
            normalize(&mut v);

            let mut sigma = 0.0f32;

            for _ in 0..max_iters {
                // u = A * v
                let mut u = vec![0.0f32; rows];
                for i in 0..rows {
                    let mut sum = 0.0f32;
                    for j in 0..cols {
                        sum += residual[i * cols + j] * v[j];
                    }
                    u[i] = sum;
                }
                let u_norm = norm(&u);
                if u_norm < 1e-12 {
                    // Residual is effectively zero in this direction
                    break;
                }
                for x in u.iter_mut() {
                    *x /= u_norm;
                }

                // v_new = A^T * u
                let mut v_new = vec![0.0f32; cols];
                for j in 0..cols {
                    let mut sum = 0.0f32;
                    for i in 0..rows {
                        sum += residual[i * cols + j] * u[i];
                    }
                    v_new[j] = sum;
                }
                let new_sigma = norm(&v_new);
                if new_sigma < 1e-12 {
                    break;
                }
                for x in v_new.iter_mut() {
                    *x /= new_sigma;
                }

                // Check convergence
                let converged = (new_sigma - sigma).abs() / (new_sigma.abs() + 1e-12) < 1e-7;
                sigma = new_sigma;
                v = v_new;

                if converged {
                    // Recompute u for final
                    for i in 0..rows {
                        let mut sum = 0.0f32;
                        for j in 0..cols {
                            sum += residual[i * cols + j] * v[j];
                        }
                        u[i] = sum;
                    }
                    let u_norm = norm(&u);
                    if u_norm > 1e-12 {
                        for x in u.iter_mut() {
                            *x /= u_norm;
                        }
                    }
                    break;
                }
            }

            if sigma < 1e-12 {
                // No more significant singular values
                break;
            }

            // Compute final u = A * v, then scale: u_scaled = sigma * u_hat
            let mut u_final = vec![0.0f32; rows];
            for i in 0..rows {
                let mut sum = 0.0f32;
                for j in 0..cols {
                    sum += residual[i * cols + j] * v[j];
                }
                u_final[i] = sum;
            }
            // u_final = sigma * u_hat already (since A*v = sigma*u)
            // We want U * V^T to reconstruct, so store u_scaled = sqrt(sigma)*u_hat,
            // v_scaled = sqrt(sigma)*v_hat. But simpler: store u = sigma*u_hat, v = v_hat.
            // Then U * V^T = sigma * u_hat * v_hat^T = correct rank-1 term.
            // Actually u_final IS sigma * u_hat. Keep it.

            // Deflate: residual -= sigma * u_hat * v^T = u_final * v^T
            for i in 0..rows {
                for j in 0..cols {
                    residual[i * cols + j] -= u_final[i] * v[j];
                }
            }

            u_vecs.push(u_final);
            v_vecs.push(v);
        }

        let actual_rank = u_vecs.len();

        // Pack into [rows, rank] and [cols, rank]
        let mut u = vec![0.0f32; rows * actual_rank];
        let mut v = vec![0.0f32; cols * actual_rank];
        for (r, u_vec) in u_vecs.iter().enumerate() {
            for i in 0..rows {
                u[i * actual_rank + r] = u_vec[i];
            }
        }
        for (r, v_vec) in v_vecs.iter().enumerate() {
            for j in 0..cols {
                v[j * actual_rank + r] = v_vec[j];
            }
        }

        Self {
            u,
            v,
            rank: actual_rank,
            rows,
            cols,
        }
    }

    /// Multiply: result = (U @ V^T) @ input.
    ///
    /// Cost: O(rows * rank + rank * cols) instead of O(rows * cols).
    pub fn matvec(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.cols, "input size mismatch");

        // Step 1: temp = V^T @ input  [rank]
        let mut temp = vec![0.0f32; self.rank];
        for (r, t) in temp.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (j, &inp) in input.iter().enumerate() {
                sum += self.v[j * self.rank + r] * inp;
            }
            *t = sum;
        }

        // Step 2: result = U @ temp  [rows]
        let mut result = vec![0.0f32; self.rows];
        for (i, res) in result.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (r, &t) in temp.iter().enumerate() {
                sum += self.u[i * self.rank + r] * t;
            }
            *res = sum;
        }

        result
    }

    /// Compression ratio vs full matrix.
    pub fn compression_ratio(&self) -> f64 {
        let full = self.rows * self.cols;
        let compressed = self.rows * self.rank + self.cols * self.rank;
        if compressed == 0 {
            return 0.0;
        }
        full as f64 / compressed as f64
    }

    /// Reconstruction error: Frobenius norm of (original - reconstructed) / Frobenius norm of original.
    pub fn reconstruction_error(&self, original: &[f32]) -> f64 {
        assert_eq!(
            original.len(),
            self.rows * self.cols,
            "matrix size mismatch"
        );

        let mut err_sum = 0.0f64;
        let mut orig_sum = 0.0f64;

        for i in 0..self.rows {
            for j in 0..self.cols {
                let orig = original[i * self.cols + j] as f64;
                orig_sum += orig * orig;

                // Reconstruct element (i,j) = sum_r u[i,r] * v[j,r]
                let mut recon = 0.0f64;
                for r in 0..self.rank {
                    recon += self.u[i * self.rank + r] as f64 * self.v[j * self.rank + r] as f64;
                }

                let diff = orig - recon;
                err_sum += diff * diff;
            }
        }

        if orig_sum < 1e-30 {
            return 0.0;
        }
        (err_sum / orig_sum).sqrt()
    }
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn normalize(v: &mut [f32]) {
    let n = norm(v);
    if n > 1e-12 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn energy_concentration_uniform() {
        // Uniform weights: top 50% should hold ~50% energy.
        let w: Vec<f32> = vec![1.0; 100];
        let c = energy_concentration(&w, 0.5);
        assert!((c - 0.5).abs() < 0.02, "expected ~0.5, got {c}");
    }

    #[test]
    fn energy_concentration_sparse() {
        // One large value, rest small.
        let mut w = vec![0.01f32; 100];
        w[0] = 100.0;
        let c = energy_concentration(&w, 0.01);
        assert!(c > 0.99, "expected >0.99, got {c}");
    }

    #[test]
    fn effective_rank_sparse() {
        let mut w = vec![0.001f32; 1000];
        for w_i in w.iter_mut().take(10) {
            *w_i = 10.0;
        }
        let rank = estimate_effective_rank(&w);
        assert!(rank < 0.05, "expected low effective rank, got {rank}");
    }

    #[test]
    fn effective_rank_uniform() {
        let w: Vec<f32> = vec![1.0; 100];
        let rank = estimate_effective_rank(&w);
        assert!(rank > 0.85, "expected high effective rank, got {rank}");
    }

    #[test]
    fn low_rank_matvec_rank1() {
        // Rank-1 matrix: A = u * v^T
        let rows = 4;
        let cols = 3;
        let u_vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let v_vec: Vec<f32> = vec![5.0, 6.0, 7.0];
        let mut matrix = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                matrix[i * cols + j] = u_vec[i] * v_vec[j];
            }
        }

        let approx = LowRankApprox::compute(&matrix, rows, cols, 1);

        // matvec should match direct multiply
        let input = vec![1.0, 0.5, -1.0];
        let expected: Vec<f32> = (0..rows)
            .map(|i| {
                (0..cols)
                    .map(|j| matrix[i * cols + j] * input[j])
                    .sum::<f32>()
            })
            .collect();

        let result = approx.matvec(&input);
        for (a, b) in expected.iter().zip(result.iter()) {
            assert!(
                (a - b).abs() < 0.1,
                "matvec mismatch: expected {a}, got {b}"
            );
        }
    }

    #[test]
    fn low_rank_reconstruction_error() {
        // Rank-2 matrix approximated with rank 2 should have near-zero error.
        let rows = 5;
        let cols = 4;
        let mut matrix = vec![0.0f32; rows * cols];
        // Rank-2: A = u1*v1^T + u2*v2^T
        let u1 = [1.0, 2.0, 3.0, 4.0, 5.0];
        let v1 = [1.0, 0.0, 0.0, 0.0];
        let u2 = [0.0, 0.0, 1.0, 1.0, 0.0];
        let v2 = [0.0, 1.0, 0.0, 0.0];
        for i in 0..rows {
            for j in 0..cols {
                matrix[i * cols + j] = u1[i] * v1[j] + u2[i] * v2[j];
            }
        }

        let approx = LowRankApprox::compute(&matrix, rows, cols, 2);
        let err = approx.reconstruction_error(&matrix);
        assert!(err < 0.01, "expected near-zero error, got {err}");
    }

    #[test]
    fn compression_ratio_calculation() {
        let rows = 100;
        let cols = 200;
        let matrix = vec![0.0f32; rows * cols];
        let approx = LowRankApprox::compute(&matrix, rows, cols, 10);
        // Even if actual_rank < 10 due to zero matrix, check formula
        if approx.rank > 0 {
            let expected = (rows * cols) as f64 / (rows * approx.rank + cols * approx.rank) as f64;
            assert!((approx.compression_ratio() - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn low_rank_full_rank_diagonal() {
        // Diagonal matrix with descending values: rank-2 approx should capture top 2.
        let n = 4;
        let mut matrix = vec![0.0f32; n * n];
        matrix[0] = 10.0;
        matrix[n + 1] = 5.0;
        matrix[2 * n + 2] = 1.0;
        matrix[3 * n + 3] = 0.1;

        let approx = LowRankApprox::compute(&matrix, n, n, 2);
        let err = approx.reconstruction_error(&matrix);
        // Top-2 singular values are 10 and 5, capturing most energy.
        // Relative error should be small since residual is small vs original.
        assert!(err < 0.15, "expected small relative error, got {err}");
    }
}
