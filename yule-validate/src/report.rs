use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Category {
    Correctness,
    Performance,
    Infrastructure,
}

impl fmt::Display for Category {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Category::Correctness => write!(f, "CORRECT"),
            Category::Performance => write!(f, "PERF"),
            Category::Infrastructure => write!(f, "INFRA"),
        }
    }
}

pub struct TestResult {
    pub id: u32,
    pub name: String,
    pub category: Category,
    pub passed: bool,
    pub message: String,
    pub duration_ms: f64,
    pub metrics: Option<HashMap<String, f64>>,
}

impl TestResult {
    pub fn pass(id: u32, name: &str, category: Category, message: &str, duration_ms: f64) -> Self {
        Self {
            id,
            name: name.to_string(),
            category,
            passed: true,
            message: message.to_string(),
            duration_ms,
            metrics: None,
        }
    }

    pub fn fail(id: u32, name: &str, category: Category, message: &str, duration_ms: f64) -> Self {
        Self {
            id,
            name: name.to_string(),
            category,
            passed: false,
            message: message.to_string(),
            duration_ms,
            metrics: None,
        }
    }

    pub fn with_metrics(mut self, metrics: HashMap<String, f64>) -> Self {
        self.metrics = Some(metrics);
        self
    }
}

pub fn print_report(results: &[TestResult]) {
    println!();
    println!(
        "{:<4} {:<6} {:<8} {:<45} {:>8}  DETAILS",
        "ID", "STAT", "CAT", "TEST", "TIME"
    );
    println!("{:-<120}", "");

    for r in results {
        let status = if r.passed { "PASS" } else { "FAIL" };
        let status_color = if r.passed { "\x1b[32m" } else { "\x1b[31m" };
        let reset = "\x1b[0m";

        println!(
            "{:<4} {status_color}{:<6}{reset} {:<8} {:<45} {:>7.1}ms  {}",
            r.id, status, r.category, r.name, r.duration_ms, r.message
        );

        if let Some(ref metrics) = r.metrics {
            for (k, v) in metrics {
                println!("     {:>60}: {:.2}", k, v);
            }
        }
    }

    println!("{:-<120}", "");

    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = total - passed;
    let metric_count: usize = results
        .iter()
        .filter_map(|r| r.metrics.as_ref())
        .map(|m| m.len())
        .sum();

    if failed == 0 {
        println!("\x1b[32m{passed}/{total} tests passed.\x1b[0m {metric_count} metrics collected.");
    } else {
        println!(
            "\x1b[31m{passed}/{total} tests passed, {failed} FAILED.\x1b[0m {metric_count} metrics collected."
        );
    }
    println!();
}

pub fn print_json(results: &[TestResult]) {
    let entries: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            let mut obj = serde_json::json!({
                "id": r.id,
                "name": r.name,
                "category": r.category.to_string(),
                "passed": r.passed,
                "message": r.message,
                "duration_ms": r.duration_ms,
            });
            if let Some(ref metrics) = r.metrics {
                obj["metrics"] = serde_json::json!(metrics);
            }
            obj
        })
        .collect();

    let output = serde_json::json!({
        "total": results.len(),
        "passed": results.iter().filter(|r| r.passed).count(),
        "failed": results.iter().filter(|r| !r.passed).count(),
        "results": entries,
    });

    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}

pub fn compute_stats(values: &[f64]) -> (f64, f64, f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let mean = sorted.iter().sum::<f64>() / n as f64;
    let p50 = sorted[n / 2];
    let p99 = sorted[(n * 99 / 100).min(n - 1)];
    let min = sorted[0];
    let max = sorted[n - 1];
    (mean, p50, p99, min, max)
}
