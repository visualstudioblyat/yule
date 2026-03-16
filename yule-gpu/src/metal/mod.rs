pub mod device;
pub mod memory;
pub mod pipelines;

use yule_core::error::{Result, YuleError};

use crate::{BackendKind, BufferHandle, ComputeBackend, DeviceInfo};
use device::MetalDeviceWrapper;
use memory::MetalMemoryManager;
use pipelines::MetalPipelineManager;

pub struct MetalBackend {
    device: MetalDeviceWrapper,
    memory: MetalMemoryManager,
    pipelines: MetalPipelineManager,
}

impl MetalBackend {
    pub fn new() -> Result<Self> {
        let device = MetalDeviceWrapper::new()?;
        let memory = MetalMemoryManager::new();
        let pipelines = MetalPipelineManager::new(device.inner())?;

        tracing::info!(
            device = %device.name(),
            vram_mb = device.memory_bytes() / (1024 * 1024),
            "metal backend initialized"
        );

        Ok(Self {
            device,
            memory,
            pipelines,
        })
    }

    pub fn is_available() -> bool {
        MetalDeviceWrapper::is_available()
    }

    fn dispatch_simple(
        &self,
        kernel: &str,
        buffer_handles: &[&BufferHandle],
        params: &[u8],
        params_index: u64,
        grid: metal::MTLSize,
        threadgroup: metal::MTLSize,
    ) -> Result<()> {
        let pipeline = self.pipelines.get_pipeline(kernel)?;
        let cmd_buf = self.pipelines.command_queue().new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);

        let buffers = self.memory.buffers.lock().unwrap();
        for (i, handle) in buffer_handles.iter().enumerate() {
            let buf = buffers
                .get(&handle.0)
                .ok_or_else(|| YuleError::Gpu(format!("buffer {} not found", handle.0)))?;
            encoder.set_buffer(i as u64, Some(buf), 0);
        }

        encoder.set_bytes(
            params_index,
            params.len() as u64,
            params.as_ptr() as *const _,
        );

        encoder.dispatch_threadgroups(grid, threadgroup);
        encoder.end_encoding();

        drop(buffers);
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        Ok(())
    }
}

impl ComputeBackend for MetalBackend {
    fn name(&self) -> &str {
        "metal"
    }

    fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            name: self.device.name(),
            backend: BackendKind::Metal,
            memory_bytes: self.device.memory_bytes(),
            compute_units: self.device.max_threads_per_threadgroup(),
        }
    }

    fn allocate(&self, size_bytes: usize) -> Result<BufferHandle> {
        self.memory.allocate(&self.device, size_bytes)
    }

    fn free(&self, handle: BufferHandle) -> Result<()> {
        self.memory.free(handle)
    }

    fn copy_to_device(&self, data: &[u8], handle: &BufferHandle) -> Result<()> {
        self.memory.copy_to_device(data, handle)
    }

    fn copy_from_device(&self, handle: &BufferHandle, data: &mut [u8]) -> Result<()> {
        self.memory.copy_from_device(handle, data)
    }

    fn copy_buffer(&self, src: &BufferHandle, dst: &BufferHandle, size: usize) -> Result<()> {
        self.memory.copy_buffer(&self.device, src, dst, 0, 0, size)
    }

    fn copy_buffer_offset(
        &self,
        src: &BufferHandle,
        dst: &BufferHandle,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> Result<()> {
        self.memory
            .copy_buffer(&self.device, src, dst, src_offset, dst_offset, size)
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    fn matmul(
        &self,
        a: &BufferHandle,
        b: &BufferHandle,
        out: &BufferHandle,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<()> {
        #[repr(C)]
        struct MatmulParams {
            m: u32,
            n: u32,
            k: u32,
        }
        let params = MatmulParams { m, n, k };
        self.dispatch_simple(
            "matmul_kernel",
            &[a, b, out],
            &params,
            (m, 1, 1),
            (256, 1, 1),
        )
    }

    fn add(
        &self,
        a: &BufferHandle,
        b: &BufferHandle,
        output: &BufferHandle,
        size: u32,
    ) -> Result<()> {
        #[repr(C)]
        struct Params {
            size: u32,
        }
        let params = Params { size };
        let params_bytes = unsafe {
            std::slice::from_raw_parts(
                &params as *const Params as *const u8,
                std::mem::size_of::<Params>(),
            )
        };
        let wg_x = ((size + 255) / 256) as u64;
        self.dispatch_simple(
            "add_kernel",
            &[a, b, output],
            params_bytes,
            3,
            metal::MTLSize {
                width: wg_x,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        )
    }

    fn silu(&self, input: &BufferHandle, output: &BufferHandle, size: u32) -> Result<()> {
        // Fused silu_mul: gate=input, up=output, out=output
        #[repr(C)]
        struct Params {
            size: u32,
        }
        let params = Params { size };
        let params_bytes = unsafe {
            std::slice::from_raw_parts(
                &params as *const Params as *const u8,
                std::mem::size_of::<Params>(),
            )
        };
        let wg_x = ((size + 255) / 256) as u64;
        self.dispatch_simple(
            "silu_mul_kernel",
            &[input, output, output],
            params_bytes,
            3,
            metal::MTLSize {
                width: wg_x,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        )
    }

    fn element_mul(
        &self,
        a: &BufferHandle,
        b: &BufferHandle,
        output: &BufferHandle,
        size: u32,
    ) -> Result<()> {
        // Reuse silu_mul shader: gate=a, up=b, out=output
        // This applies silu to a then multiplies by b, matching the Vulkan backend pattern.
        #[repr(C)]
        struct Params {
            size: u32,
        }
        let params = Params { size };
        let params_bytes = unsafe {
            std::slice::from_raw_parts(
                &params as *const Params as *const u8,
                std::mem::size_of::<Params>(),
            )
        };
        let wg_x = ((size + 255) / 256) as u64;
        self.dispatch_simple(
            "silu_mul_kernel",
            &[a, b, output],
            params_bytes,
            3,
            metal::MTLSize {
                width: wg_x,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        )
    }

    fn rms_norm(
        &self,
        input: &BufferHandle,
        weight: &BufferHandle,
        output: &BufferHandle,
        size: u32,
        eps: f32,
    ) -> Result<()> {
        #[repr(C)]
        struct Params {
            size: u32,
            eps: f32,
        }
        let params = Params { size, eps };
        let params_bytes = unsafe {
            std::slice::from_raw_parts(
                &params as *const Params as *const u8,
                std::mem::size_of::<Params>(),
            )
        };
        // Single workgroup reduction
        self.dispatch_simple(
            "rms_norm_kernel",
            &[input, weight, output],
            params_bytes,
            3,
            metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        )
    }

    fn softmax(&self, input: &BufferHandle, output: &BufferHandle, size: u32) -> Result<()> {
        #[repr(C)]
        struct Params {
            size: u32,
        }
        let params = Params { size };
        let params_bytes = unsafe {
            std::slice::from_raw_parts(
                &params as *const Params as *const u8,
                std::mem::size_of::<Params>(),
            )
        };
        // Single workgroup reduction
        self.dispatch_simple(
            "softmax_kernel",
            &[input, output],
            params_bytes,
            2,
            metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        )
    }

    fn rope(
        &self,
        q: &BufferHandle,
        k: &BufferHandle,
        pos: u32,
        head_dim: u32,
        freq_base: f32,
        n_heads_q: u32,
        n_heads_k: u32,
    ) -> Result<()> {
        #[repr(C)]
        struct Params {
            pos: u32,
            head_dim: u32,
            freq_base_bits: u32,
            n_heads_q: u32,
            n_heads_k: u32,
        }
        let params = Params {
            pos,
            head_dim,
            freq_base_bits: freq_base.to_bits(),
            n_heads_q,
            n_heads_k,
        };
        let params_bytes = unsafe {
            std::slice::from_raw_parts(
                &params as *const Params as *const u8,
                std::mem::size_of::<Params>(),
            )
        };
        let half_dim = head_dim / 2;
        let total_threads = n_heads_q * half_dim + n_heads_k * half_dim;
        let wg_x = ((total_threads + 63) / 64) as u64;
        // RoPE modifies Q and K in-place; bind same buffers for read/write
        self.dispatch_simple(
            "rope_kernel",
            &[q, k],
            params_bytes,
            2,
            metal::MTLSize {
                width: wg_x,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        )
    }

    fn attn_score(
        &self,
        q: &BufferHandle,
        k_cache: &BufferHandle,
        scores: &BufferHandle,
        head_dim: u32,
        seq_len: u32,
        head_offset: u32,
        kv_offset: u32,
        kv_stride: u32,
    ) -> Result<()> {
        #[repr(C)]
        struct Params {
            head_dim: u32,
            seq_len: u32,
            head_offset: u32,
            kv_offset: u32,
            kv_stride: u32,
        }
        let params = Params {
            head_dim,
            seq_len,
            head_offset,
            kv_offset,
            kv_stride,
        };
        let params_bytes = unsafe {
            std::slice::from_raw_parts(
                &params as *const Params as *const u8,
                std::mem::size_of::<Params>(),
            )
        };
        // One workgroup per position
        self.dispatch_simple(
            "attn_score_kernel",
            &[q, k_cache, scores],
            params_bytes,
            3,
            metal::MTLSize {
                width: seq_len as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        )
    }

    fn attn_value(
        &self,
        weights: &BufferHandle,
        v_cache: &BufferHandle,
        output: &BufferHandle,
        head_dim: u32,
        seq_len: u32,
        kv_offset: u32,
        kv_stride: u32,
        out_offset: u32,
    ) -> Result<()> {
        #[repr(C)]
        struct Params {
            head_dim: u32,
            seq_len: u32,
            kv_offset: u32,
            kv_stride: u32,
            out_offset: u32,
        }
        let params = Params {
            head_dim,
            seq_len,
            kv_offset,
            kv_stride,
            out_offset,
        };
        let params_bytes = unsafe {
            std::slice::from_raw_parts(
                &params as *const Params as *const u8,
                std::mem::size_of::<Params>(),
            )
        };
        // One workgroup per output dimension
        self.dispatch_simple(
            "attn_value_kernel",
            &[weights, v_cache, output],
            params_bytes,
            3,
            metal::MTLSize {
                width: head_dim as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        )
    }

    fn quantized_matmul(
        &self,
        weights: &BufferHandle,
        input: &BufferHandle,
        output: &BufferHandle,
        n_rows: u32,
        n_cols: u32,
        dtype: yule_core::dtype::DType,
    ) -> Result<()> {
        let kernel_name = match dtype {
            yule_core::dtype::DType::Q4_0 => "qmv_q4_0_kernel",
            yule_core::dtype::DType::Q4_K => "qmv_q4_k_kernel",
            yule_core::dtype::DType::Q6_K => "qmv_q6_k_kernel",
            yule_core::dtype::DType::Q8_0 => "qmv_q8_0_kernel",
            _ => {
                return Err(YuleError::Gpu(format!(
                    "unsupported dtype for Metal qmv: {dtype:?}"
                )));
            }
        };

        let block_size = dtype.block_size() as u32;
        let blocks_per_row = (n_cols + block_size - 1) / block_size;

        #[repr(C)]
        struct Params {
            n_rows: u32,
            n_cols: u32,
            blocks_per_row: u32,
        }
        let params = Params {
            n_rows,
            n_cols,
            blocks_per_row,
        };
        let params_bytes = unsafe {
            std::slice::from_raw_parts(
                &params as *const Params as *const u8,
                std::mem::size_of::<Params>(),
            )
        };
        // One workgroup per output row, 256 threads each
        self.dispatch_simple(
            kernel_name,
            &[weights, input, output],
            params_bytes,
            3,
            metal::MTLSize {
                width: n_rows as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        )
    }
}
