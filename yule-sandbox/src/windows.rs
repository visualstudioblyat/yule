use crate::{Sandbox, SandboxConfig, SandboxGuard, SandboxedProcess};
use yule_core::error::{Result, YuleError};

use windows_sys::Win32::Foundation::*;
use windows_sys::Win32::System::JobObjects::*;
use windows_sys::Win32::System::Threading::*;

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

    fn spawn(&self, config: &SandboxConfig) -> Result<SandboxedProcess> {
        spawn_in_job(config)
    }
}

fn spawn_in_job(config: &SandboxConfig) -> Result<SandboxedProcess> {
    use std::os::windows::ffi::OsStrExt;
    use std::os::windows::io::{FromRawHandle, OwnedHandle};
    use std::process::{ChildStdin, ChildStdout};

    unsafe {
        // 1. Create a job object with the same restrictions as apply_job_object
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() {
            return Err(YuleError::Sandbox(format!(
                "CreateJobObjectW failed: {}",
                GetLastError()
            )));
        }

        // memory limit + kill on close + active process limit
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

        // ui restrictions
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

        // 2. Get current executable path as wide string
        let exe = std::env::current_exe()
            .map_err(|e| YuleError::Sandbox(format!("cannot determine current exe: {e}")))?;
        let exe_wide: Vec<u16> = exe
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();

        // 3. Create the child process in a suspended state
        let mut si: STARTUPINFOW = std::mem::zeroed();
        si.cb = std::mem::size_of::<STARTUPINFOW>() as u32;

        // Create pipes for stdin/stdout
        use windows_sys::Win32::Security::SECURITY_ATTRIBUTES;
        let mut sa: SECURITY_ATTRIBUTES = std::mem::zeroed();
        sa.nLength = std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32;
        sa.bInheritHandle = TRUE;
        sa.lpSecurityDescriptor = std::ptr::null_mut();

        // stdin pipe: parent writes, child reads
        let mut child_stdin_rd: HANDLE = std::ptr::null_mut();
        let mut child_stdin_wr: HANDLE = std::ptr::null_mut();
        if windows_sys::Win32::System::Pipes::CreatePipe(
            &mut child_stdin_rd,
            &mut child_stdin_wr,
            &sa,
            0,
        ) == 0
        {
            CloseHandle(job);
            return Err(YuleError::Sandbox(format!(
                "CreatePipe (stdin) failed: {}",
                GetLastError()
            )));
        }
        // Prevent the write end of stdin from being inherited
        SetHandleInformation(child_stdin_wr, HANDLE_FLAG_INHERIT, 0);

        // stdout pipe: child writes, parent reads
        let mut child_stdout_rd: HANDLE = std::ptr::null_mut();
        let mut child_stdout_wr: HANDLE = std::ptr::null_mut();
        if windows_sys::Win32::System::Pipes::CreatePipe(
            &mut child_stdout_rd,
            &mut child_stdout_wr,
            &sa,
            0,
        ) == 0
        {
            CloseHandle(job);
            CloseHandle(child_stdin_rd);
            CloseHandle(child_stdin_wr);
            return Err(YuleError::Sandbox(format!(
                "CreatePipe (stdout) failed: {}",
                GetLastError()
            )));
        }
        // Prevent the read end of stdout from being inherited
        SetHandleInformation(child_stdout_rd, HANDLE_FLAG_INHERIT, 0);

        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput = child_stdin_rd;
        si.hStdOutput = child_stdout_wr;
        si.hStdError = windows_sys::Win32::System::Console::GetStdHandle(
            windows_sys::Win32::System::Console::STD_ERROR_HANDLE,
        );

        let mut pi: PROCESS_INFORMATION = std::mem::zeroed();

        let ok = CreateProcessW(
            exe_wide.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            TRUE, // inherit handles
            CREATE_SUSPENDED,
            std::ptr::null(),
            std::ptr::null(),
            &si,
            &mut pi,
        );
        if ok == 0 {
            let err = GetLastError();
            CloseHandle(job);
            CloseHandle(child_stdin_rd);
            CloseHandle(child_stdin_wr);
            CloseHandle(child_stdout_rd);
            CloseHandle(child_stdout_wr);
            return Err(YuleError::Sandbox(format!(
                "CreateProcessW failed: {err}"
            )));
        }

        // Close the child-side pipe ends in the parent
        CloseHandle(child_stdin_rd);
        CloseHandle(child_stdout_wr);

        // 4. Assign the suspended process to the job object
        let ok = AssignProcessToJobObject(job, pi.hProcess);
        if ok == 0 {
            let err = GetLastError();
            // Kill the suspended process since we can't sandbox it
            TerminateProcess(pi.hProcess, 1);
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
            CloseHandle(job);
            CloseHandle(child_stdin_wr);
            CloseHandle(child_stdout_rd);
            return Err(YuleError::Sandbox(format!(
                "AssignProcessToJobObject failed: {err}"
            )));
        }

        // 5. Resume the process now that it's in the job
        ResumeThread(pi.hThread);

        let pid = pi.dwProcessId;

        // Close handles we no longer need (thread handle, process handle)
        // The job handle keeps the process alive via KILL_ON_JOB_CLOSE
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        // Note: we intentionally do not close the job handle here. It must
        // stay open so KILL_ON_JOB_CLOSE terminates the child when the parent
        // exits. The OS reclaims the handle when the parent process exits.

        // Wrap the raw HANDLEs into Rust std types
        let stdin_handle = OwnedHandle::from_raw_handle(child_stdin_wr);
        let stdout_handle = OwnedHandle::from_raw_handle(child_stdout_rd);

        let stdin: ChildStdin = std::process::ChildStdin::from(stdin_handle);
        let stdout: ChildStdout = std::process::ChildStdout::from(stdout_handle);

        tracing::info!(
            pid,
            memory_limit = config.max_memory_bytes,
            "sandboxed child process spawned in job object"
        );

        Ok(SandboxedProcess { pid, stdin, stdout })
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
