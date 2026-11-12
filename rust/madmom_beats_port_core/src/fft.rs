// SPDX-License-Identifier: BSD-2-Clause

//! Radix-2 FFT of a real frame in double precision. Twiddle factors come
//! from the `libm` crate and the operation order is fixed, so every platform
//! computes the same bits.

pub(crate) struct RealFft {
    /// `cos(2 pi j / n)` and `-sin(2 pi j / n)` for `j < n / 2`.
    cos: Vec<f64>,
    sin: Vec<f64>,
    /// The same factors in the order the butterfly passes read them: for each
    /// pass over blocks of `size`, entries `k * n / size` for `k < size / 2`.
    pass_cos: Vec<f64>,
    pass_sin: Vec<f64>,
    /// Bit-reversed position of each element of the half-size FFT.
    order: Vec<usize>,
    re: Vec<f64>,
    im: Vec<f64>,
}

impl RealFft {
    /// `n` is a power of two of at least 4 (config validation).
    pub(crate) fn new(n: usize) -> Self {
        let m = n / 2;
        let angle = |j: usize| 2.0 * std::f64::consts::PI * j as f64 / n as f64;
        let bits = m.trailing_zeros();
        let cos: Vec<f64> = (0..m).map(|j| libm::cos(angle(j))).collect();
        let sin: Vec<f64> = (0..m).map(|j| -libm::sin(angle(j))).collect();
        let pass_index = (1..=bits).flat_map(|p| (0..1usize << (p - 1)).map(move |k| k * (n >> p)));
        Self {
            pass_cos: pass_index.clone().map(|j| cos[j]).collect(),
            pass_sin: pass_index.map(|j| sin[j]).collect(),
            cos,
            sin,
            order: (0..m)
                .map(|j| {
                    j.reverse_bits()
                        .checked_shr(usize::BITS - bits)
                        .unwrap_or(0)
                })
                .collect(),
            re: vec![0.0; m],
            im: vec![0.0; m],
        }
    }

    /// Bins `0..n / 2` of the spectrum of `frame` (length `n`), rounded to
    /// single precision as madmom stores them (`complex64`).
    pub(crate) fn process(&mut self, frame: &[f64], out: &mut [(f32, f32)]) {
        let m = self.re.len();
        // The even samples are the real parts and the odd samples the
        // imaginary parts of one complex FFT of half the length.
        for (j, &pos) in self.order.iter().enumerate() {
            self.re[pos] = frame[2 * j];
            self.im[pos] = frame[2 * j + 1];
        }
        let mut half = 1;
        while half < m {
            let (pc, ps) = (
                &self.pass_cos[half - 1..2 * half - 1],
                &self.pass_sin[half - 1..2 * half - 1],
            );
            let blocks = self
                .re
                .chunks_exact_mut(2 * half)
                .zip(self.im.chunks_exact_mut(2 * half));
            for (re, im) in blocks {
                let (ra, rb) = re.split_at_mut(half);
                let (ia, ib) = im.split_at_mut(half);
                for k in 0..half {
                    let (wr, wi) = (pc[k], ps[k]);
                    let tr = wr * rb[k] - wi * ib[k];
                    let ti = wr * ib[k] + wi * rb[k];
                    rb[k] = ra[k] - tr;
                    ib[k] = ia[k] - ti;
                    ra[k] += tr;
                    ia[k] += ti;
                }
            }
            half *= 2;
        }
        // Split the half-size result into the spectra of the even and odd
        // samples, then combine them into the spectrum of the frame.
        for (k, bin) in out.iter_mut().enumerate().take(m) {
            let c = (m - k) % m;
            let (zr, zi, cr, ci) = (self.re[k], self.im[k], self.re[c], -self.im[c]);
            let (er, ei) = ((zr + cr) * 0.5, (zi + ci) * 0.5);
            let (or, oi) = ((zi - ci) * 0.5, (cr - zr) * 0.5);
            let (wr, wi) = (self.cos[k], self.sin[k]);
            *bin = (
                (er + (wr * or - wi * oi)) as f32,
                (ei + (wr * oi + wi * or)) as f32,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RealFft;

    #[test]
    fn matches_direct_dft() {
        for n in [2usize, 4, 8, 64, 1024] {
            let frame: Vec<f64> = (0..n)
                .map(|i| ((i * 7919) % 101) as f64 / 50.0 - 1.0)
                .collect();
            let mut out = vec![(0.0f32, 0.0f32); n / 2];
            RealFft::new(n).process(&frame, &mut out);
            for (k, &(re, im)) in out.iter().enumerate() {
                let (mut er, mut ei) = (0.0f64, 0.0f64);
                for (i, &x) in frame.iter().enumerate() {
                    let a = -2.0 * std::f64::consts::PI * (k * i % n) as f64 / n as f64;
                    er += x * a.cos();
                    ei += x * a.sin();
                }
                let tol = 1e-5 * (1.0 + er.hypot(ei));
                assert!(
                    (re as f64 - er).abs() <= tol && (im as f64 - ei).abs() <= tol,
                    "n {n} bin {k}"
                );
            }
        }
    }
}
