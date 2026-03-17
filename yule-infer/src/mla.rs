//! Multi-Latent Attention (MLA) KV cache compression.
//! Compresses K/V vectors into a low-rank latent representation.
//! Reference: DeepSeek-V2 (2024) — 93.3% KV cache reduction.

/// MLA compressor that projects K/V into latent space.
pub struct MlaCompressor {
    kv_dim: usize,
    latent_dim: usize,
    /// Down-projection: kv_dim -> latent_dim. Shape: [latent_dim, kv_dim] row-major.
    down_proj: Vec<f32>,
    /// Up-projection: latent_dim -> kv_dim. Shape: [kv_dim, latent_dim] row-major.
    up_proj: Vec<f32>,
}

impl MlaCompressor {
    /// Create from explicit projection matrices.
    ///
    /// `down_proj` must have length `latent_dim * kv_dim` (row-major [latent_dim, kv_dim]).
    /// `up_proj` must have length `kv_dim * latent_dim` (row-major [kv_dim, latent_dim]).
    pub fn new(kv_dim: usize, latent_dim: usize, down_proj: Vec<f32>, up_proj: Vec<f32>) -> Self {
        assert!(
            latent_dim > 0 && latent_dim <= kv_dim,
            "latent_dim must be in (0, kv_dim]"
        );
        assert_eq!(
            down_proj.len(),
            latent_dim * kv_dim,
            "down_proj shape mismatch"
        );
        assert_eq!(up_proj.len(), kv_dim * latent_dim, "up_proj shape mismatch");
        Self {
            kv_dim,
            latent_dim,
            down_proj,
            up_proj,
        }
    }

    /// Create from SVD of sample K/V data.
    ///
    /// Takes sample K/V vectors (flattened, each of length `kv_dim`), computes a
    /// truncated eigendecomposition of the sample covariance via power iteration,
    /// and uses the top singular vectors as projection matrices.
    pub fn from_samples(kv_dim: usize, latent_dim: usize, samples: &[f32]) -> Self {
        assert!(latent_dim > 0 && latent_dim <= kv_dim);
        assert_eq!(
            samples.len() % kv_dim,
            0,
            "samples must be a multiple of kv_dim"
        );
        let n_samples = samples.len() / kv_dim;
        assert!(n_samples > 0, "need at least one sample");

        // Compute covariance matrix C = (1/n) * X^T X  where X is [n_samples, kv_dim].
        // Then extract top-k eigenvectors via power iteration (deflation).
        let mut covariance = vec![0.0f32; kv_dim * kv_dim];
        for s in 0..n_samples {
            let row = &samples[s * kv_dim..(s + 1) * kv_dim];
            for i in 0..kv_dim {
                for j in i..kv_dim {
                    let val = row[i] * row[j];
                    covariance[i * kv_dim + j] += val;
                    if i != j {
                        covariance[j * kv_dim + i] += val;
                    }
                }
            }
        }
        let inv_n = 1.0 / n_samples as f32;
        for v in covariance.iter_mut() {
            *v *= inv_n;
        }

        // Power iteration with deflation to extract top eigenvectors.
        let eigenvectors = power_iteration_topk(&covariance, kv_dim, latent_dim, 100);

        // down_proj = eigenvectors^T packed as [latent_dim, kv_dim]
        // (each eigenvector is a row of down_proj)
        let down_proj = eigenvectors.clone();

        // up_proj = transpose of down_proj = [kv_dim, latent_dim]
        let mut up_proj = vec![0.0f32; kv_dim * latent_dim];
        for i in 0..latent_dim {
            for j in 0..kv_dim {
                up_proj[j * latent_dim + i] = down_proj[i * kv_dim + j];
            }
        }

        Self {
            kv_dim,
            latent_dim,
            down_proj,
            up_proj,
        }
    }

    /// Compress a K or V vector: kv_dim -> latent_dim.
    pub fn compress(&self, kv: &[f32]) -> Vec<f32> {
        assert_eq!(kv.len(), self.kv_dim, "input dimension mismatch");
        let mut latent = vec![0.0f32; self.latent_dim];
        for i in 0..self.latent_dim {
            let mut sum = 0.0f32;
            for j in 0..self.kv_dim {
                sum += self.down_proj[i * self.kv_dim + j] * kv[j];
            }
            latent[i] = sum;
        }
        latent
    }

    /// Decompress latent vector: latent_dim -> kv_dim.
    pub fn decompress(&self, latent: &[f32]) -> Vec<f32> {
        assert_eq!(latent.len(), self.latent_dim, "latent dimension mismatch");
        let mut kv = vec![0.0f32; self.kv_dim];
        for i in 0..self.kv_dim {
            let mut sum = 0.0f32;
            for j in 0..self.latent_dim {
                sum += self.up_proj[i * self.latent_dim + j] * latent[j];
            }
            kv[i] = sum;
        }
        kv
    }

    /// Compression ratio (original dim / latent dim).
    pub fn compression_ratio(&self) -> f64 {
        self.kv_dim as f64 / self.latent_dim as f64
    }

    /// Latent dimension.
    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    /// Original KV dimension.
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }
}

/// MLA-compressed KV cache.
///
/// Stores K/V vectors in compressed latent form. On write, vectors are projected
/// down; on read, they are projected back up. This trades compute for memory.
pub struct MlaKvCache {
    compressor: MlaCompressor,
    num_layers: u32,
    max_seq_len: u32,
    current_len: u32,
    /// Latent K storage: one Vec per layer, each `[max_seq_len * latent_dim]`.
    k_latent: Vec<Vec<f32>>,
    /// Latent V storage: one Vec per layer, each `[max_seq_len * latent_dim]`.
    v_latent: Vec<Vec<f32>>,
}

impl MlaKvCache {
    /// Create a new MLA-compressed KV cache.
    pub fn new(compressor: MlaCompressor, num_layers: u32, max_seq_len: u32) -> Self {
        let latent_dim = compressor.latent_dim();
        let layer_size = max_seq_len as usize * latent_dim;
        let k_latent = (0..num_layers).map(|_| vec![0.0f32; layer_size]).collect();
        let v_latent = (0..num_layers).map(|_| vec![0.0f32; layer_size]).collect();
        Self {
            compressor,
            num_layers,
            max_seq_len,
            current_len: 0,
            k_latent,
            v_latent,
        }
    }

    /// Write a K/V pair at the current position for a given layer.
    /// The K and V vectors are compressed before storage.
    pub fn write_kv(&mut self, layer: u32, k: &[f32], v: &[f32]) {
        assert!((layer as u32) < self.num_layers, "layer out of range");
        assert!(self.current_len < self.max_seq_len, "cache full");

        let latent_k = self.compressor.compress(k);
        let latent_v = self.compressor.compress(v);

        let ld = self.compressor.latent_dim();
        let offset = self.current_len as usize * ld;
        let layer = layer as usize;

        self.k_latent[layer][offset..offset + ld].copy_from_slice(&latent_k);
        self.v_latent[layer][offset..offset + ld].copy_from_slice(&latent_v);
    }

    /// Advance the sequence position after writing all layers for one token.
    pub fn advance(&mut self) {
        self.current_len += 1;
    }

    /// Read and decompress the K vector at a given position and layer.
    pub fn read_k(&self, layer: u32, pos: u32) -> Vec<f32> {
        assert!((layer as u32) < self.num_layers, "layer out of range");
        assert!(pos < self.current_len, "position out of range");

        let ld = self.compressor.latent_dim();
        let offset = pos as usize * ld;
        let latent = &self.k_latent[layer as usize][offset..offset + ld];
        self.compressor.decompress(latent)
    }

    /// Read and decompress the V vector at a given position and layer.
    pub fn read_v(&self, layer: u32, pos: u32) -> Vec<f32> {
        assert!((layer as u32) < self.num_layers, "layer out of range");
        assert!(pos < self.current_len, "position out of range");

        let ld = self.compressor.latent_dim();
        let offset = pos as usize * ld;
        let latent = &self.v_latent[layer as usize][offset..offset + ld];
        self.compressor.decompress(latent)
    }

    /// Total size in bytes of the compressed cache storage.
    pub fn size_bytes(&self) -> usize {
        let per_layer = self.max_seq_len as usize * self.compressor.latent_dim() * 4; // f32
        per_layer * 2 * self.num_layers as usize // K + V
    }

    /// Size that an equivalent uncompressed cache would occupy.
    pub fn uncompressed_size_bytes(&self) -> usize {
        let per_layer = self.max_seq_len as usize * self.compressor.kv_dim() * 4;
        per_layer * 2 * self.num_layers as usize
    }

    /// Current sequence length.
    pub fn current_len(&self) -> u32 {
        self.current_len
    }

    /// Clear the cache, resetting the sequence position.
    pub fn clear(&mut self) {
        self.current_len = 0;
        for buf in self.k_latent.iter_mut().chain(self.v_latent.iter_mut()) {
            buf.fill(0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Extract the top-k eigenvectors of a symmetric matrix via power iteration
/// with deflation. Returns a flat vec of shape [k, dim] (row-major).
fn power_iteration_topk(matrix: &[f32], dim: usize, k: usize, max_iters: usize) -> Vec<f32> {
    let mut deflated = matrix.to_vec();
    let mut eigenvectors = Vec::with_capacity(k * dim);

    for _ in 0..k {
        // Start with a non-zero vector
        let mut v = vec![0.0f32; dim];
        for (i, val) in v.iter_mut().enumerate() {
            // Deterministic seed: use index
            *val = ((i as f32 + 1.0) * 0.31415).sin();
        }
        normalize(&mut v);

        for _ in 0..max_iters {
            // w = A * v
            let mut w = vec![0.0f32; dim];
            for i in 0..dim {
                let mut sum = 0.0f32;
                for j in 0..dim {
                    sum += deflated[i * dim + j] * v[j];
                }
                w[i] = sum;
            }
            normalize(&mut w);

            // Check convergence (dot product close to 1)
            let dot: f32 = v.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
            v = w;
            if dot.abs() > 1.0 - 1e-7 {
                break;
            }
        }

        // Compute eigenvalue: lambda = v^T A v
        let mut lambda = 0.0f32;
        for i in 0..dim {
            let mut row_sum = 0.0f32;
            for j in 0..dim {
                row_sum += deflated[i * dim + j] * v[j];
            }
            lambda += v[i] * row_sum;
        }

        // Deflate: A <- A - lambda * v * v^T
        for i in 0..dim {
            for j in 0..dim {
                deflated[i * dim + j] -= lambda * v[i] * v[j];
            }
        }

        eigenvectors.extend_from_slice(&v);
    }

    eigenvectors
}

/// Normalize a vector in-place. If the norm is zero, leave unchanged.
fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an orthogonal compressor from an identity-like sub-block for testing.
    fn make_identity_compressor(kv_dim: usize, latent_dim: usize) -> MlaCompressor {
        // down_proj: [latent_dim, kv_dim] — first latent_dim rows of identity
        let mut down = vec![0.0f32; latent_dim * kv_dim];
        for i in 0..latent_dim {
            down[i * kv_dim + i] = 1.0;
        }
        // up_proj: [kv_dim, latent_dim] — transpose
        let mut up = vec![0.0f32; kv_dim * latent_dim];
        for i in 0..latent_dim {
            up[i * latent_dim + i] = 1.0;
        }
        MlaCompressor::new(kv_dim, latent_dim, down, up)
    }

    #[test]
    fn compress_decompress_identity_roundtrip() {
        let kv_dim = 8;
        let latent_dim = 4;
        let comp = make_identity_compressor(kv_dim, latent_dim);

        // A vector whose energy is concentrated in the first 4 dims should roundtrip exactly.
        let kv = vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        let latent = comp.compress(&kv);
        assert_eq!(latent.len(), latent_dim);
        let recovered = comp.decompress(&latent);
        assert_eq!(recovered.len(), kv_dim);
        for (a, b) in kv.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-6, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn compression_ratio() {
        let comp = make_identity_compressor(128, 16);
        assert!((comp.compression_ratio() - 8.0).abs() < 1e-6);
    }

    #[test]
    fn from_samples_recovers_principal_directions() {
        // Samples that lie in a 2D subspace of R^4.
        let kv_dim = 4;
        let latent_dim = 2;
        let samples: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, // along e1
            0.0, 1.0, 0.0, 0.0, // along e2
            2.0, 0.0, 0.0, 0.0, // along e1
            0.0, 3.0, 0.0, 0.0, // along e2
            1.0, 1.0, 0.0, 0.0, // in e1-e2 plane
        ];
        let comp = MlaCompressor::from_samples(kv_dim, latent_dim, &samples);

        // A vector in the e1-e2 plane should roundtrip with low error.
        let kv = vec![3.0, 4.0, 0.0, 0.0];
        let recovered = comp.decompress(&comp.compress(&kv));
        let err: f32 = kv
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        assert!(err < 0.1, "roundtrip error too large: {err}");
    }

    #[test]
    fn kv_cache_write_read_roundtrip() {
        let kv_dim = 8;
        let latent_dim = 8; // no compression loss for this test
        let comp = make_identity_compressor(kv_dim, latent_dim);
        let mut cache = MlaKvCache::new(comp, 2, 16);

        let k = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        cache.write_kv(0, &k, &v);
        cache.write_kv(1, &k, &v);
        cache.advance();

        let k_out = cache.read_k(0, 0);
        let v_out = cache.read_v(0, 0);
        for (a, b) in k.iter().zip(k_out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        for (a, b) in v.iter().zip(v_out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn kv_cache_size_reduction() {
        let kv_dim = 128;
        let latent_dim = 16;
        let comp = make_identity_compressor(kv_dim, latent_dim);
        let cache = MlaKvCache::new(comp, 32, 4096);

        let compressed = cache.size_bytes();
        let uncompressed = cache.uncompressed_size_bytes();
        let ratio = uncompressed as f64 / compressed as f64;
        assert!(ratio > 7.0, "expected ~8x compression, got {ratio:.1}x");
    }

    #[test]
    fn kv_cache_clear() {
        let comp = make_identity_compressor(4, 4);
        let mut cache = MlaKvCache::new(comp, 1, 8);
        let k = vec![1.0; 4];
        let v = vec![2.0; 4];
        cache.write_kv(0, &k, &v);
        cache.advance();
        assert_eq!(cache.current_len(), 1);
        cache.clear();
        assert_eq!(cache.current_len(), 0);
    }
}
