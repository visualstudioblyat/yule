use std::sync::{Arc, Barrier, Mutex};
use std::thread;

/// Lightweight thread pool for parallel matmul. Workers stay alive between calls.
/// Uses a barrier for synchronization instead of channel overhead.
pub struct ThreadPool {
    workers: Vec<thread::JoinHandle<()>>,
    work: Arc<WorkQueue>,
    num_threads: usize,
}

struct WorkQueue {
    // Each round: main thread sets tasks, signals start barrier,
    // workers execute, signal done barrier.
    tasks: Mutex<Vec<Box<dyn FnOnce() + Send>>>,
    start: Barrier,
    done: Barrier,
    shutdown: std::sync::atomic::AtomicBool,
}

impl ThreadPool {
    pub fn new(num_threads: usize) -> Self {
        let num_threads = num_threads.max(1);

        let work = Arc::new(WorkQueue {
            tasks: Mutex::new(Vec::new()),
            start: Barrier::new(num_threads + 1), // +1 for main thread
            done: Barrier::new(num_threads + 1),
            shutdown: std::sync::atomic::AtomicBool::new(false),
        });

        let mut workers = Vec::with_capacity(num_threads);
        for _ in 0..num_threads {
            let w = Arc::clone(&work);
            workers.push(thread::spawn(move || {
                loop {
                    // Wait for work
                    w.start.wait();

                    if w.shutdown.load(std::sync::atomic::Ordering::Relaxed) {
                        w.done.wait();
                        break;
                    }

                    // Grab a task
                    let task = w.tasks.lock().unwrap().pop();
                    if let Some(f) = task {
                        f();
                    }

                    // Signal completion
                    w.done.wait();
                }
            }));
        }

        Self {
            workers,
            work,
            num_threads,
        }
    }

    /// Execute tasks in parallel. Blocks until all complete.
    pub fn execute<F>(&self, tasks: Vec<F>)
    where
        F: FnOnce() + Send + 'static,
    {
        {
            let mut queue = self.work.tasks.lock().unwrap();
            queue.clear();
            for t in tasks {
                queue.push(Box::new(t));
            }
        }

        // Signal workers to start
        self.work.start.wait();
        // Wait for all workers to finish
        self.work.done.wait();
    }

    pub fn num_threads(&self) -> usize {
        self.num_threads
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.work
            .shutdown
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.work.start.wait(); // release workers
        self.work.done.wait(); // wait for them to see shutdown

        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn test_basic_execution() {
        let pool = ThreadPool::new(4);
        let counter = Arc::new(AtomicU32::new(0));

        let mut tasks: Vec<Box<dyn FnOnce() + Send>> = Vec::new();
        for _ in 0..4 {
            let c = Arc::clone(&counter);
            tasks.push(Box::new(move || {
                c.fetch_add(1, Ordering::Relaxed);
            }));
        }

        pool.execute(tasks);
        assert_eq!(counter.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn test_multiple_rounds() {
        let pool = ThreadPool::new(2);
        let counter = Arc::new(AtomicU32::new(0));

        for _ in 0..10 {
            let c = Arc::clone(&counter);
            let tasks: Vec<Box<dyn FnOnce() + Send>> = vec![Box::new(move || {
                c.fetch_add(1, Ordering::Relaxed);
            })];
            pool.execute(tasks);
        }

        assert_eq!(counter.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn test_drop_cleans_up() {
        let pool = ThreadPool::new(4);
        assert_eq!(pool.num_threads(), 4);
        drop(pool); // should not hang
    }
}
