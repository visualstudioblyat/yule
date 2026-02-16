use crate::{Sandbox, SandboxConfig, SandboxGuard, SandboxedProcess};
use yule_core::error::{Result, YuleError};

use windows_sys::Win32::Foundation::*;
use windows_sys::Win32::System::JobObjects::*;
use windows_sys::Win32::System::Threading::GetCurrentProcess;

pub struct WindowsSandbox;

impl WindowsSandbox {
    pub fn new() -> Self {
        Self
    }
}

impl Default for WindowsSandbox {
    fn default() -> Self {
        Self::new()
    }
}

impl Sandbox for WindowsSandbox {
    fn apply_to_current_process(&self, config: &SandboxConfig) -> Result<SandboxGuard> {
        apply_job_object(config)
    }

    fn spawn(&self, _config: &SandboxConfig) -> Result<SandboxedProcess> {
        Err(YuleError::Sandbox(
            "broker-target spawn not yet implemented".into(),
        ))
    }
}

pub(crate) unsafe fn close_job_handle(handle: *mut std::ffi::c_void) {
    unsafe {
        CloseHandle(handle);
    }
}

fn apply_job_object(config: &SandboxConfig) -> Result<SandboxGuard> {
    unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() {
            return Err(YuleError::Sandbox(format!(
                "CreateJobObjectW failed: {}",
                GetLastError()
            )));
        }

        // memory limit + kill on close + no child processes
        let mut ext_info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        ext_info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY
            | JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        ext_info.ProcessMemoryLimit = config.max_memory_bytes as usize;
        ext_info.BasicLimitInformation.ActiveProcessLimit = 1;

        let ok = SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &ext_info as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        );
        if ok == 0 {
            CloseHandle(job);
            return Err(YuleError::Sandbox(format!(
                "SetInformationJobObject (limits) failed: {}",
                GetLastError()
            )));
        }

        // ui restrictions — block clipboard, desktop, display settings
        let mut ui: JOBOBJECT_BASIC_UI_RESTRICTIONS = std::mem::zeroed();
        ui.UIRestrictionsClass = JOB_OBJECT_UILIMIT_DESKTOP
            | JOB_OBJECT_UILIMIT_DISPLAYSETTINGS
            | JOB_OBJECT_UILIMIT_EXITWINDOWS
            | JOB_OBJECT_UILIMIT_GLOBALATOMS
            | JOB_OBJECT_UILIMIT_HANDLES
            | JOB_OBJECT_UILIMIT_READCLIPBOARD
            | JOB_OBJECT_UILIMIT_SYSTEMPARAMETERS
            | JOB_OBJECT_UILIMIT_WRITECLIPBOARD;

        let ok = SetInformationJobObject(
            job,
            JobObjectBasicUIRestrictions,
            &ui as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<JOBOBJECT_BASIC_UI_RESTRICTIONS>() as u32,
        );
        if ok == 0 {
            CloseHandle(job);
            return Err(YuleError::Sandbox(format!(
                "SetInformationJobObject (ui) failed: {}",
                GetLastError()
            )));
        }

        // assign current process to the job
        let process = GetCurrentProcess();
        let ok = AssignProcessToJobObject(job, process);
        if ok == 0 {
            let err = GetLastError();
            CloseHandle(job);
            if err == ERROR_ACCESS_DENIED {
                return Err(YuleError::Sandbox(
                    "process already in a job object (nested jobs require Windows 8+)".into(),
                ));
            }
            return Err(YuleError::Sandbox(format!(
                "AssignProcessToJobObject failed: {err}"
            )));
        }

        tracing::info!(
            memory_limit = config.max_memory_bytes,
            "sandbox applied: job object with memory limit and ui restrictions"
        );

        Ok(SandboxGuard { job_handle: job })
    }
}
