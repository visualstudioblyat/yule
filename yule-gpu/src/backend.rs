use crate::BackendKind;

pub fn select_best_backend(available: &[BackendKind]) -> BackendKind {
    // priority: CUDA > Metal > Vulkan > CPU
    for &preferred in &[
        BackendKind::Cuda,
        BackendKind::Metal,
        BackendKind::Vulkan,
        BackendKind::Cpu,
    ] {
        if available.contains(&preferred) {
            return preferred;
        }
    }
    BackendKind::Cpu
}
