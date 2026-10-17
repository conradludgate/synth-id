use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::distr::Distribution;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use synth_id::SynthId;

const CTX: usize = 4;
const LAYER_COUNTS: [u8; 6] = [0, 1, 2, 4, 6, 8];

/// A concentrated next-token distribution: probability mass on 20 scattered
/// token ids with a Zipf-like (1/rank) decay — closer to a real LLM's
/// post-softmax top-k than a flat uniform over the whole vocabulary. The
/// tournament therefore samples with heavy collisions, as it would in practice.
const K: usize = 20;

struct TopK {
    tokens: [u32; K],
    cumulative: [f64; K],
}

impl TopK {
    fn new() -> Self {
        let mut tokens = [0u32; K];
        let mut cumulative = [0.0f64; K];
        let total: f64 = (0..K).map(|i| 1.0 / (i as f64 + 1.0)).sum();
        let mut acc = 0.0;
        for i in 0..K {
            tokens[i] = i as u32 * 2_500 + 7; // scatter across ~0..50k
            acc += (1.0 / (i as f64 + 1.0)) / total; // normalized Zipf weight
            cumulative[i] = acc;
        }
        Self { tokens, cumulative }
    }
}

impl Distribution<u32> for TopK {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u32 {
        let x: f64 = rng.random();
        let idx = self.cumulative.iter().position(|&c| x < c).unwrap_or(K - 1);
        self.tokens[idx]
    }
}

// How much does the watermark tournament cost over a plain draw?
//   single_sample     — one `dist.sample`, i.e. generation with no watermark
//   samples_only/<L>   — drawing the 2^L candidates, without the tournament
//   tournament/<L>     — full `generate`: 2^L draws + the scoring tournament
// tournament vs single_sample is the end-to-end overhead; tournament vs
// samples_only isolates the PRF/tournament work from the raw sampling.
fn tournament(c: &mut Criterion) {
    let dist = TopK::new();
    let mut group = c.benchmark_group("generate");

    group.bench_function("single_sample", |b| {
        let mut rng = StdRng::seed_from_u64(1);
        b.iter(|| dist.sample(&mut rng));
    });

    for layers in LAYER_COUNTS {
        group.bench_with_input(
            BenchmarkId::new("samples_only", layers),
            &layers,
            |b, &layers| {
                let mut rng = StdRng::seed_from_u64(1);
                let n = 1usize << layers;
                b.iter(|| {
                    let mut acc = 0u32;
                    for _ in 0..n {
                        acc ^= dist.sample(&mut rng);
                    }
                    acc
                });
            },
        );

        let synth = SynthId::new([7u8; 32], CTX, layers);
        group.bench_with_input(BenchmarkId::new("tournament", layers), &layers, |b, _| {
            let mut rng = StdRng::seed_from_u64(1);
            let mut cx = synth.context();
            b.iter(|| cx.generate(&dist, &mut rng));
        });
    }

    group.finish();
}

criterion_group!(benches, tournament);
criterion_main!(benches);
