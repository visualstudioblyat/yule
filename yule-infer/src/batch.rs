use std::collections::VecDeque;

use crate::SamplingParams;
use crate::kv_cache::SequenceState;

pub struct BatchScheduler {
    active_sequences: Vec<ActiveSequence>,
    pending_requests: VecDeque<InferenceRequest>,
    max_batch_size: usize,
    max_total_tokens: usize,
}

pub struct ActiveSequence {
    pub seq_id: u64,
    pub state: SequenceState,
    pub generated_tokens: Vec<u32>,
    pub max_new_tokens: u32,
    pub is_prefilling: bool,
    pub is_complete: bool,
}

pub struct InferenceRequest {
    pub seq_id: u64,
    pub prompt_tokens: Vec<u32>,
    pub max_new_tokens: u32,
    pub sampling: SamplingParams,
    pub eos_token: Option<u32>,
}

#[allow(dead_code)] // used by callers assembling batch output
pub struct BatchResult {
    pub completed: Vec<(u64, Vec<u32>)>,
}

impl BatchScheduler {
    pub fn new(max_batch_size: usize, max_total_tokens: usize) -> Self {
        Self {
            active_sequences: Vec::new(),
            pending_requests: VecDeque::new(),
            max_batch_size,
            max_total_tokens,
        }
    }

    /// Submit a new inference request.
    pub fn submit(&mut self, request: InferenceRequest) {
        self.pending_requests.push_back(request);
    }

    /// Select the next batch of sequences to process.
    ///
    /// Promotes pending requests to active sequences up to the batch size and
    /// total-token budget, then returns mutable references to all active
    /// (non-complete) sequences that fit in the batch.
    pub fn schedule(&mut self) -> Vec<&mut ActiveSequence> {
        // Admit pending requests while budget allows
        while self.active_sequences.len() < self.max_batch_size {
            let total_tokens: usize = self
                .active_sequences
                .iter()
                .map(|s| s.state.token_count as usize + s.generated_tokens.len())
                .sum();

            if let Some(req) = self.pending_requests.front() {
                let new_tokens = req.prompt_tokens.len();
                if total_tokens + new_tokens > self.max_total_tokens {
                    break;
                }
            } else {
                break;
            }

            let req = self.pending_requests.pop_front().unwrap();
            let num_layers = 1; // page tables are managed externally; placeholder
            let state = SequenceState {
                seq_id: req.seq_id,
                page_tables: vec![Vec::new(); num_layers],
                token_count: req.prompt_tokens.len() as u32,
                max_tokens: req.prompt_tokens.len() as u32 + req.max_new_tokens,
            };
            self.active_sequences.push(ActiveSequence {
                seq_id: req.seq_id,
                state,
                generated_tokens: Vec::new(),
                max_new_tokens: req.max_new_tokens,
                is_prefilling: true,
                is_complete: false,
            });
        }

        self.active_sequences
            .iter_mut()
            .filter(|s| !s.is_complete)
            .collect()
    }

    /// Mark a sequence as complete.
    pub fn complete(&mut self, seq_id: u64) {
        for seq in &mut self.active_sequences {
            if seq.seq_id == seq_id {
                seq.is_complete = true;
                break;
            }
        }
    }

    /// Check if there are pending or active (non-complete) sequences.
    pub fn has_work(&self) -> bool {
        !self.pending_requests.is_empty() || self.active_sequences.iter().any(|s| !s.is_complete)
    }

    /// Drain completed sequences and return their results.
    pub fn drain_completed(&mut self) -> Vec<(u64, Vec<u32>)> {
        let mut results = Vec::new();
        self.active_sequences.retain(|seq| {
            if seq.is_complete {
                results.push((seq.seq_id, seq.generated_tokens.clone()));
                false
            } else {
                true
            }
        });
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_request(seq_id: u64, prompt_len: usize, max_new: u32) -> InferenceRequest {
        InferenceRequest {
            seq_id,
            prompt_tokens: vec![1; prompt_len],
            max_new_tokens: max_new,
            sampling: SamplingParams::default(),
            eos_token: None,
        }
    }

    #[test]
    fn test_batch_scheduler_submit_and_schedule() {
        let mut sched = BatchScheduler::new(4, 1024);
        assert!(!sched.has_work());

        sched.submit(make_request(1, 10, 20));
        sched.submit(make_request(2, 15, 20));
        sched.submit(make_request(3, 5, 10));
        assert!(sched.has_work());

        let batch = sched.schedule();
        assert_eq!(batch.len(), 3);

        // Verify all sequences are active and prefilling
        for seq in &batch {
            assert!(seq.is_prefilling);
            assert!(!seq.is_complete);
        }

        // Pending queue should be empty now
        assert!(sched.pending_requests.is_empty());
    }

    #[test]
    fn test_batch_scheduler_respects_max_batch_size() {
        let mut sched = BatchScheduler::new(2, 1024);

        sched.submit(make_request(1, 10, 20));
        sched.submit(make_request(2, 10, 20));
        sched.submit(make_request(3, 10, 20));

        let batch = sched.schedule();
        // Only 2 should be admitted
        assert_eq!(batch.len(), 2);
        // One request should remain pending
        assert_eq!(sched.pending_requests.len(), 1);
    }

    #[test]
    fn test_batch_scheduler_complete_and_drain() {
        let mut sched = BatchScheduler::new(4, 1024);

        sched.submit(make_request(1, 10, 20));
        sched.submit(make_request(2, 15, 20));
        let _ = sched.schedule();

        // Simulate some generation
        sched.active_sequences[0].generated_tokens = vec![100, 101, 102];
        sched.active_sequences[1].generated_tokens = vec![200, 201];

        // Complete seq 1
        sched.complete(1);
        assert!(sched.has_work()); // seq 2 still active

        let results = sched.drain_completed();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
        assert_eq!(results[0].1, vec![100, 101, 102]);

        // Seq 1 should be removed from active
        assert_eq!(sched.active_sequences.len(), 1);
        assert_eq!(sched.active_sequences[0].seq_id, 2);

        // Complete seq 2
        sched.complete(2);
        let results = sched.drain_completed();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 2);

        assert!(!sched.has_work());
    }
}
