use std::cmp;

use rand::Rng;
use rand::distr::Distribution;
use sha2::block_api::compress256;

const PAD: u32 = u32::MAX;
const TOKEN_WIDTH: usize = std::mem::size_of::<u32>();
const PREFIX_CAPACITY: usize = 192;
const MAX_CTX: usize = (PREFIX_CAPACITY - 32) / TOKEN_WIDTH;

pub struct SynthId {
    /// Key for the PRF
    key: [u8; 32],
    /// How many prior tokens to use in the seed
    ctx: usize,
    /// How many layers the tournament should have
    layers: u8,
}

impl SynthId {
    /// Build a watermarker from a PRF `key`, the number of prior tokens mixed
    /// into each seed (`ctx`), and the number of tournament `layers` (each
    /// generation step plays `2^layers` sampled candidates). `ctx` must be at
    /// most 40 so the optimized SHA-256 prefix remains on the stack.
    pub fn new(key: [u8; 32], ctx: usize, layers: u8) -> Self {
        assert!(ctx <= MAX_CTX, "ctx must be at most {MAX_CTX}");
        Self { key, ctx, layers }
    }

    /// Open an empty rolling context for streaming generation. Repeated
    /// [`SynthIdCtx::generate`] calls seeds from the tokens produced so far and
    /// rolls the newest one into the window — the same window `check` walks.
    pub fn context(&self) -> SynthIdCtx<'_> {
        SynthIdCtx {
            synth: self,
            buf: vec![PAD; (2 * self.ctx).saturating_sub(1)],
            pos: 0,
        }
    }

    /// Open a rolling context primed with an existing token history; only the
    /// most recent `ctx` tokens end up in the window.
    pub fn context_from(&self, prior: &[u32]) -> SynthIdCtx<'_> {
        let mut cx = self.context();
        for &t in &prior[prior.len().saturating_sub(self.ctx)..] {
            cx.advance(t);
        }
        cx
    }

    pub fn check(&self, tokens: impl Iterator<Item = u32>) -> impl Iterator<Item = u8> {
        let layers = self.layers;
        let mut cx = self.context();

        tokens.map(move |t| {
            let seed = cx.seed();
            let s: u8 = (0..layers).map(|l| seed.score(t, l)).sum();
            cx.advance(t);
            s
        })
    }
}

/// A rolling window over the most recent tokens, borrowed from a [`SynthId`].
/// Drives streaming generation (and backs [`SynthId::check`]) so the caller
/// doesn't have to track the context slice by hand.
pub struct SynthIdCtx<'a> {
    synth: &'a SynthId,
    /// Double-width ring buffer of length `2*ctx - 1` (empty when `ctx == 0`).
    /// The most recent `ctx` tokens are always the contiguous slice
    /// `buf[pos..pos + ctx]`; each token is written twice — at `pos` and its
    /// mirror `pos + ctx` — so the window stays contiguous as `pos` cycles,
    /// making [`Self::advance`] O(1) instead of shifting the whole window.
    buf: Vec<u32>,
    pos: usize,
}

impl SynthIdCtx<'_> {
    /// Sample the next token — biased by the watermark for the current window —
    /// and roll it into the window.
    pub fn generate<D: Distribution<u32>, R: Rng>(&mut self, dist: &D, rng: &mut R) -> u32 {
        let seed = self.seed();
        let t = tree_reduce(self.synth.layers, dist, rng, |token, layer| {
            seed.score(token, layer)
        });
        self.advance(t);
        t
    }

    fn window(&self) -> &[u32] {
        &self.buf[self.pos..self.pos + self.synth.ctx]
    }

    /// Precompute the tournament seed: hash `key ‖ window` (zero-padded to whole
    /// SHA-256 blocks) into a reusable state, so each `score` only compresses one
    /// more block holding `layer ‖ token`.
    fn seed(&self) -> Seed {
        let window = self.window();
        // key ‖ window, zero-padded up to a whole number of 64-byte blocks.
        let prefix_len = (32 + window.len() * TOKEN_WIDTH).next_multiple_of(64);
        let mut prefix = [0u8; PREFIX_CAPACITY];
        prefix[..32].copy_from_slice(&self.synth.key);
        let mut off = 32;
        for t in window {
            prefix[off..off + TOKEN_WIDTH].copy_from_slice(&t.to_le_bytes());
            off += TOKEN_WIDTH;
        }

        let mut state = SHA256_INIT;
        for block in prefix[..prefix_len].chunks_exact(64) {
            compress(&mut state, block.try_into().expect("chunk is 64 bytes"));
        }

        // Final-block template: layer ‖ token ‖ SHA-256 padding, where the length
        // is the *total* message (prefix + layer + token) in bits.
        let total_bits = (prefix_len + 1 + TOKEN_WIDTH) as u64 * 8;
        let mut block = [0u8; 64];
        block[1 + TOKEN_WIDTH] = 0x80;
        block[56..64].copy_from_slice(&total_bits.to_be_bytes());

        Seed { state, block }
    }

    fn advance(&mut self, t: u32) {
        if self.buf.is_empty() {
            return; // ctx == 0: no window to maintain
        }
        self.buf[self.pos] = t;
        let mirror = self.pos + self.synth.ctx;
        if mirror < self.buf.len() {
            self.buf[mirror] = t;
        }
        self.pos = (self.pos + 1) % self.synth.ctx;
    }
}

/// SHA-256 initial hash state.
const SHA256_INIT: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

fn compress(state: &mut [u32; 8], block: &[u8; 64]) {
    compress256(state, std::slice::from_ref(block));
}

/// A precomputed tournament seed: the SHA-256 state after `key ‖ window`, plus a
/// template for the final `layer ‖ token` block. [`Seed::score`] folds one
/// token/layer onto it with a single block compression — no re-hashing of
/// `key ‖ window`, no block buffer.
struct Seed {
    state: [u32; 8],
    block: [u8; 64],
}

impl Seed {
    fn score(&self, token: u32, layer: u8) -> u8 {
        let mut block = self.block;
        block[0] = layer;
        block[1..1 + TOKEN_WIDTH].copy_from_slice(&token.to_le_bytes());
        let mut state = self.state;
        compress(&mut state, &block);
        // The bit is arbitrary as long as generate and check agree; take the LSB
        // of state word 0.
        (state[0] & 1) as u8
    }
}

/// Sample `2^levels` leaves from `dist` and reduce them through a balanced
/// binary tournament, returning the winner. At each match the higher-scoring
/// token advances, ties broken by a coin flip; `score_at(token, layer)` supplies
/// the per-layer watermark bit. `rng` drives both leaf sampling and tie-breaks,
/// so the whole draw is reproducible from a seeded generator.
fn tree_reduce<D, R>(levels: u8, dist: &D, rng: &mut R, score_at: impl Fn(u32, u8) -> u8) -> u32
where
    D: Distribution<u32>,
    R: Rng,
{
    // One slot per height: `slots[h]` parks a pending subtree of height `h`,
    // with `PAD` (no real token equals it) marking an empty slot. Each new leaf
    // is a binary-counter increment — carry upward, merging with the occupied
    // slot at each height until an empty one. The carry reaches height `levels`
    // after exactly `2^levels` leaves, so we stop there. Fixed `levels + 1`
    // length ⇒ the box is allocated once and can never reallocate.
    let top = levels as usize;
    let mut slots: Box<[u32]> = vec![PAD; top + 1].into_boxed_slice();

    loop {
        if slots[top] != PAD {
            return slots[top];
        }
        let mut node = dist.sample(rng);
        let mut height = 0;
        while slots[height] != PAD {
            let left = slots[height];
            // Identical tokens have identical scores, so this match cannot
            // change the winner and needs no hashing or tie-break draw.
            if left != node {
                node = match score_at(left, height as u8).cmp(&score_at(node, height as u8)) {
                    cmp::Ordering::Greater => left,
                    cmp::Ordering::Less => node,
                    cmp::Ordering::Equal if rng.random_ratio(1, 2) => left,
                    cmp::Ordering::Equal => node,
                };
            }
            slots[height] = PAD;
            height += 1;
        }
        slots[height] = node;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::distr::Uniform;
    use rand::rngs::StdRng;
    use sha2::{Digest, Sha256};

    fn synth() -> SynthId {
        SynthId::new([42u8; 32], 4, 4)
    }

    // --- `score` ---------------------------------------------------------

    #[test]
    fn score_is_a_single_bit() {
        let seed = synth().context().seed();
        for t in 0..2000u32 {
            for l in 0..8u8 {
                let s = seed.score(t, l);
                assert!(s == 0 || s == 1, "score must be 0 or 1, got {s}");
            }
        }
    }

    #[test]
    fn score_depends_on_token_layer_and_key() {
        let seed1 = SynthId::new([1u8; 32], 4, 4).context().seed();
        let seed2 = SynthId::new([2u8; 32], 4, 4).context().seed();

        // The bit should vary with layer (and key), not be constant.
        let varies_by_layer: Vec<u8> = (0..64u8).map(|l| seed1.score(7u32, l)).collect();
        assert!(varies_by_layer.contains(&0));
        assert!(varies_by_layer.contains(&1));

        let differs_by_key = (0..256u32)
            .filter(|&t| seed1.score(t, 0) != seed2.score(t, 0))
            .count();
        assert!(
            differs_by_key > 0,
            "different keys should produce different bits for some token"
        );
    }

    #[test]
    fn compress_seed_matches_plain_sha256() {
        // Validate the hand-built compress256 blocks against a plain Sha256 over
        // the same layout: key ‖ window ‖ zero-filler ‖ layer ‖ token.
        let key = [42u8; 32]; // same key as `synth()`
        let window = [11u32, 22, 33, 44];
        let seed = synth().context_from(&window).seed();

        let prefix_len = (32 + window.len() * 4).next_multiple_of(64);
        for token in [0u32, 1, 7, 1000, 123_456] {
            for layer in 0..8u8 {
                let mut msg = Vec::new();
                msg.extend_from_slice(&key);
                for w in &window {
                    msg.extend_from_slice(&w.to_le_bytes());
                }
                msg.resize(prefix_len, 0); // zero filler to a block boundary
                msg.push(layer);
                msg.extend_from_slice(&token.to_le_bytes());

                // state[0] & 1 == LSB of word 0 == low bit of big-endian digest byte 3.
                let expected = Sha256::digest(&msg)[3] & 1;
                assert_eq!(
                    seed.score(token, layer),
                    expected,
                    "mismatch at token {token} layer {layer}"
                );
            }
        }
    }

    // --- `tree_reduce` ---------------------------------------------------

    #[test]
    fn tree_reduce_zero_levels_is_a_single_sample() {
        use std::cell::RefCell;

        // Height 0 = one sampled leaf, no matches (so `score_at` is never asked).
        let calls = RefCell::new(0u32);
        let dist = Uniform::new(10u32, 20).unwrap();
        let mut rng = StdRng::seed_from_u64(1);
        let out = tree_reduce(0, &dist, &mut rng, |_t, _l| {
            *calls.borrow_mut() += 1;
            0
        });
        assert!((10..20).contains(&out), "single sampled leaf");
        assert_eq!(calls.into_inner(), 0, "no matches at level 0");
    }

    #[test]
    fn tree_reduce_builds_a_balanced_tree() {
        use std::cell::RefCell;

        // A height-3 tournament has 4 matches at layer 0, 2 at layer 1, 1 at
        // layer 2. `score_at` is consulted for both sides of every match, so the
        // per-layer call counts are twice those: 8 / 4 / 2. (A degenerate
        // left-leaning tree would put all matches — hence all calls — at layer 0.)
        let seen = RefCell::new(Vec::new());
        let dist = Uniform::new(0u32, 1000).unwrap();
        let mut rng = StdRng::seed_from_u64(2);
        tree_reduce(3, &dist, &mut rng, |_t, l| {
            seen.borrow_mut().push(l);
            0
        });

        let seen = seen.into_inner();
        let count = |target: u8| seen.iter().filter(|&&l| l == target).count();
        assert_eq!(count(0), 8, "leaf round");
        assert_eq!(count(1), 4, "semifinals");
        assert_eq!(count(2), 2, "final");
        assert_eq!(seen.len(), 14, "2 * (2^levels - 1) score lookups");
    }

    #[test]
    fn tree_reduce_skips_matches_with_identical_tokens() {
        use std::cell::RefCell;

        struct Constant;

        impl Distribution<u32> for Constant {
            fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> u32 {
                7
            }
        }

        let calls = RefCell::new(0u32);
        let mut rng = StdRng::seed_from_u64(3);
        let out = tree_reduce(4, &Constant, &mut rng, |_token, _layer| {
            *calls.borrow_mut() += 1;
            0
        });

        assert_eq!(out, 7);
        assert_eq!(
            calls.into_inner(),
            0,
            "identical-token matches need no scores"
        );
    }

    // --- `check` ---------------------------------------------------------

    #[test]
    fn check_is_deterministic() {
        let s = synth();
        let tokens: Vec<u32> = (0..128).collect();
        let a: Vec<u8> = s.check(tokens.iter().copied()).collect();
        let b: Vec<u8> = s.check(tokens.iter().copied()).collect();
        assert_eq!(a, b);
    }

    #[test]
    fn check_yields_one_score_per_token() {
        let s = synth();
        let tokens: Vec<u32> = (0..37).collect();
        let scores: Vec<u8> = s.check(tokens.iter().copied()).collect();
        assert_eq!(scores.len(), tokens.len());
    }

    #[test]
    fn check_scores_are_bounded_by_layers() {
        let s = synth();
        for score in s.check((0..1000).collect::<Vec<u32>>().into_iter()) {
            assert!(
                score <= s.layers,
                "score {score} exceeds layers {}",
                s.layers
            );
        }
    }

    #[test]
    fn check_is_context_sensitive() {
        // The seed is keyed by the preceding context.
        let s = synth();
        let seed_a = s.context_from(&[10u32, 20, 30]).seed();
        let seed_b = s.context_from(&[40u32, 50, 60]).seed();
        assert_ne!(
            seed_a.state, seed_b.state,
            "expected the seed to depend on its context"
        );
    }

    #[test]
    fn rolling_window_matches_naive_reference() {
        // The O(1) double-width ring buffer must expose exactly the window a
        // naive "pad the front, keep the last `ctx`" reference would — checked
        // across many wrap-arounds. (Generation and detection both read through
        // `window()`, so only an independent reference can catch a bug here.)
        let s = synth();
        let n = s.ctx;

        let mut cx = s.context();
        let mut history: Vec<u32> = vec![PAD; n];

        assert_eq!(cx.window(), &history[history.len() - n..], "initial window");

        for t in 100..140u32 {
            cx.advance(t);
            history.push(t);
            assert_eq!(
                cx.window(),
                &history[history.len() - n..],
                "window diverged after pushing {t}"
            );
        }
    }

    // --- edge cases ------------------------------------------------------

    #[test]
    fn zero_layers_scores_are_all_zero() {
        let s = SynthId::new([1u8; 32], 4, 0);
        assert!(
            s.check((0..100).collect::<Vec<u32>>().into_iter())
                .all(|x| x == 0)
        );
    }

    #[test]
    fn generate_with_zero_layers_returns_a_sampled_token() {
        let s = SynthId::new([1u8; 32], 4, 0);
        let dist = Uniform::new(10u32, 20).unwrap();
        let mut rng = StdRng::seed_from_u64(4);
        for _ in 0..100 {
            let t = s.context_from(&[]).generate(&dist, &mut rng);
            assert!(
                (10..20).contains(&t),
                "token {t} outside distribution support"
            );
        }
    }

    // --- `generate` ------------------------------------------------------

    #[test]
    fn generate_returns_token_within_distribution_support() {
        let s = synth();
        let dist = Uniform::new(100u32, 200).unwrap();
        let mut rng = StdRng::seed_from_u64(5);
        let mut cx = s.context();
        for _ in 0..50 {
            let t = cx.generate(&dist, &mut rng);
            assert!(
                (100..200).contains(&t),
                "token {t} outside distribution support"
            );
        }
    }

    #[test]
    fn generated_tokens_are_biased_across_all_layers() {
        // The winner climbs from a leaf to the root, winning one match per
        // layer, so it tends to hold the "1" bit at every layer. Each match it
        // wins gives ~0.75 probability of bit 1 (it only carries a 0 up when
        // both sides were 0 and it won the coin flip), for an expected score of
        // ~0.75 * layers. An unwatermarked token averages 0.5 * layers.
        let s = synth();
        let dist = Uniform::new(0u32, 50_000).unwrap();
        let ctx = [1u32, 2, 3];
        let seed = s.context_from(&ctx).seed();
        let mut rng = StdRng::seed_from_u64(6);

        let trials = 400;
        let total: u32 = (0..trials)
            .map(|_| {
                let t = s.context_from(&ctx).generate(&dist, &mut rng);
                (0..s.layers).map(|l| seed.score(t, l) as u32).sum::<u32>()
            })
            .sum();
        let mean = total as f64 / trials as f64;

        let unwatermarked = 0.5 * f64::from(s.layers);
        assert!(
            mean > unwatermarked + 0.5,
            "expected bias across all layers; mean {mean:.3} vs unwatermarked {unwatermarked:.3}"
        );
    }

    #[test]
    fn watermarked_text_scores_higher_than_random() {
        let s = synth();
        let dist = Uniform::new(0u32, 50_000).unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        let n = 2000usize;

        // Build a watermarked sequence with a rolling context — the same window
        // `check` walks when it scores the tokens back.
        let watermarked: Vec<u32> = {
            let mut cx = s.context();
            (0..n).map(|_| cx.generate(&dist, &mut rng)).collect()
        };

        // A control sequence with no watermark: independent draws.
        let random: Vec<u32> = (0..n).map(|_| dist.sample(&mut rng)).collect();

        let wm_mean =
            s.check(watermarked.into_iter()).map(u64::from).sum::<u64>() as f64 / n as f64;
        let rnd_mean = s.check(random.into_iter()).map(u64::from).sum::<u64>() as f64 / n as f64;

        // The layer-0 bias lifts the watermarked mean by ~0.5 per token; a
        // 0.25 margin is comfortably inside the noise floor for n = 2000.
        assert!(
            wm_mean > rnd_mean + 0.25,
            "watermark not detectable: watermarked mean {wm_mean:.3} vs random mean {rnd_mean:.3}"
        );
    }
}
