// SPDX-License-Identifier: BSD-2-Clause AND BSD-3-Clause
// The tanh table and algorithm follow NumPy's float32 tanh
// (numpy/_core/src/umath/loops_hyperbolic.dispatch.c.src, BSD 3-Clause,
// Copyright (c) 2005-2025 NumPy Developers). See NOTICE.

//! NumPy/SciPy/BLAS float operations in madmom's order, via `libm` with unfused multiply-adds, so every
//! platform gets the same bits.

/// Dot product summed from index 0 upwards, starting at zero: the order BLAS
/// (`sgemv`/`sgemm`) uses for madmom's `np.dot` calls.
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (&x, &y) in a.iter().zip(b) {
        sum += x * y;
    }
    sum
}

/// `exp` for `float32`, computed in double precision and rounded.
#[inline]
pub(crate) fn exp(x: f32) -> f32 {
    libm::exp(x as f64) as f32
}

/// Natural logarithm for `float32`, computed in double precision and rounded.
#[inline]
pub(crate) fn ln(x: f32) -> f32 {
    libm::log(x as f64) as f32
}

/// `np.log10` for `float32` as madmom's reference platform (macOS arm64)
/// computes it: the double-precision result rounded, except on the inputs in
/// `LOG10_UP` and `LOG10_DOWN`, where that platform rounds one step higher or
/// lower. The tables cover every input of at least 1.
pub(crate) fn log10(x: f32) -> f32 {
    let r = libm::log10(x as f64) as f32;
    let bits = x.to_bits();
    if LOG10_UP.binary_search(&bits).is_ok() {
        f32::from_bits(r.to_bits() + 1)
    } else if LOG10_DOWN.binary_search(&bits).is_ok() {
        f32::from_bits(r.to_bits() - 1)
    } else {
        r
    }
}

const LOG10_UP: [u32; 128] = [
    0x3f8070d8, 0x3f807518, 0x3f807c70, 0x3f807e0c, 0x3f80814b, 0x3f80873d, 0x3f808d45, 0x3f808d7e,
    0x3f80965e, 0x3f809a56, 0x3f809c47, 0x3f809ce3, 0x3f809d77, 0x3f809fa6, 0x3f80a3c3, 0x3f80ae36,
    0x3f80b19b, 0x3f80b2e9, 0x3f80b34d, 0x3f80b61b, 0x3f80b952, 0x3f80ba71, 0x3f80be0e, 0x3f80c0ea,
    0x3f80c143, 0x3f80c18d, 0x3f80c3c7, 0x3f80c5c3, 0x3f80c6df, 0x3f80c883, 0x3f80cde4, 0x3f80d004,
    0x3f80dadf, 0x3f80dbc9, 0x3f80e3b7, 0x3f80e3ea, 0x3f81011d, 0x3f810da1, 0x3f81e2e9, 0x3f81e755,
    0x3f829ea4, 0x3f82a965, 0x3f82c4f9, 0x3f82cf84, 0x3f82d90d, 0x3f82da8c, 0x3f83d866, 0x3f83dcd8,
    0x3f83f04a, 0x3f84d98f, 0x3f867e33, 0x3f869869, 0x3f8699bd, 0x3f86aef6, 0x3f86c7c0, 0x3f86e852,
    0x3f889cb4, 0x3f88b64d, 0x3f8af1b3, 0x3f8cd666, 0x3f8e9758, 0x3f8eb622, 0x3f8edac1, 0x3f94b052,
    0x3f94c21d, 0x3f94d375, 0x3f9cc0a8, 0x3fa0d5bc, 0x3fa0e48f, 0x3fa3c059, 0x3fa3ef9e, 0x3fa5b4ae,
    0x3fa7d523, 0x3fa8e80e, 0x3fa8f991, 0x3faaf444, 0x3fb2f72b, 0x3fbcf5f6, 0x3fc4f4e3, 0x3fdafdb5,
    0x3fecdc7b, 0x401ad0df, 0x4022dd0e, 0x40358b42, 0x40677936, 0x40aaf8b3, 0x40b5b1d9, 0x40d0f6c6,
    0x40dda0f2, 0x40e5b7fe, 0x41ccf680, 0x421ef70b, 0x4220c589, 0x4288d7ad, 0x43079cce, 0x4391bcaa,
    0x4431ca41, 0x4451b838, 0x44887214, 0x4492a959, 0x4590c113, 0x4610e765, 0x47bcf401, 0x48c8efbc,
    0x4bace3d4, 0x4baed6d4, 0x4dff4ddc, 0x4e5af952, 0x5038f940, 0x51a2f04d, 0x5230ffe2, 0x5238b4f3,
    0x5374f20f, 0x539082ba, 0x5498f5e5, 0x54fdbb27, 0x550ee837, 0x56f8c876, 0x5708f105, 0x604df02c,
    0x62a6c1dd, 0x65716a23, 0x679adb5b, 0x6da8a20d, 0x703ae116, 0x74bfb209, 0x7926c956, 0x7c32d7c2,
];
const LOG10_DOWN: [u32; 121] = [
    0x3f800659, 0x3f8007c6, 0x3f8008d0, 0x3f800a78, 0x3f800aa2, 0x3f800eae, 0x3f800ec3, 0x3f80121c,
    0x3f801253, 0x3f8018fe, 0x3f801bbc, 0x3f801d47, 0x3f8020de, 0x3f802a1e, 0x3f802da1, 0x3f802ff2,
    0x3f803b73, 0x3f8040bb, 0x3f8041d5, 0x3f804df5, 0x3f80512c, 0x3f80ffb6, 0x3f812189, 0x3f812a07,
    0x3f81374a, 0x3f813dd7, 0x3f813dfb, 0x3f814004, 0x3f8158a0, 0x3f815ca0, 0x3f816f5a, 0x3f8176cd,
    0x3f82245e, 0x3f8234df, 0x3f824865, 0x3f832505, 0x3f833d76, 0x3f834524, 0x3f84582a, 0x3f8527c8,
    0x3f854806, 0x3f857360, 0x3f87176b, 0x3f87635d, 0x3f89399e, 0x3f893b5a, 0x3f893ebc, 0x3f8b2e0e,
    0x3f8b3a09, 0x3f8d1966, 0x3f8d24ed, 0x3f8f4127, 0x3f8f42bc, 0x3f90415f, 0x3f91095b, 0x3f916e17,
    0x3f93553a, 0x3f964c10, 0x3f97134c, 0x3f991311, 0x3fa5147f, 0x3fa51a43, 0x3fa64147, 0x3fac1d99,
    0x3fb31b52, 0x3fb3387c, 0x3fbf46f4, 0x3fc62762, 0x3fc951d7, 0x3fd31c07, 0x3fd82271, 0x3fd83b89,
    0x3fda4b0d, 0x3fe70e66, 0x401f6fce, 0x404427e7, 0x40532d9f, 0x40702a13, 0x40ac20e8, 0x41003bee,
    0x410387e5, 0x41094689, 0x410d4a09, 0x4110513f, 0x41234035, 0x415b0e2c, 0x4181648d, 0x41a7064b,
    0x41ea90aa, 0x42015800, 0x420b164b, 0x420d2943, 0x421f7441, 0x42812b92, 0x434127c3, 0x44193c35,
    0x441b2157, 0x441c554d, 0x44975d5b, 0x44d32f1f, 0x44e04bca, 0x454a7c58, 0x4621206d, 0x469f09a2,
    0x47f07281, 0x48c42bc1, 0x4a99591b, 0x4e07469b, 0x51491852, 0x55c7116b, 0x56953970, 0x575e626b,
    0x5b2d0c6e, 0x5f3315a3, 0x655528b0, 0x65903d25, 0x6705396e, 0x692319c6, 0x6d837616, 0x6f592c3c,
    0x77171318,
];

/// Sum as NumPy's `np.sum` computes it: pairwise, with eight running sums over blocks of up to 128 values.
pub(crate) fn pairwise_sum<T>(values: &[T]) -> T
where
    T: Copy + Default + std::ops::Add<Output = T>,
{
    const BLOCK: usize = 128;
    let n = values.len();
    if n < 8 {
        values.iter().fold(T::default(), |s, &v| s + v)
    } else if n <= BLOCK {
        let mut r = [T::default(); 8];
        r.copy_from_slice(&values[..8]);
        let whole = n - n % 8;
        for chunk in values[8..whole].chunks_exact(8) {
            for (sum, &v) in r.iter_mut().zip(chunk) {
                *sum = *sum + v;
            }
        }
        let paired = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        values[whole..].iter().fold(paired, |s, &v| s + v)
    } else {
        let half = n / 2;
        let half = half - half % 8;
        pairwise_sum(&values[..half]) + pairwise_sum(&values[half..])
    }
}

/// Logistic sigmoid as SciPy's `expit` computes it in single precision.
#[inline]
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}

/// Magnitude of a complex number as NumPy's `np.abs` computes it for
/// `complex64`: the larger part times `sqrt(1 + ratio^2)`.
#[inline]
pub(crate) fn complex_abs(re: f32, im: f32) -> f32 {
    let (re, im) = (re.abs(), im.abs());
    if re.is_infinite() || im.is_infinite() {
        return f32::INFINITY;
    }
    if re.is_nan() || im.is_nan() {
        return f32::NAN;
    }
    let larger = re.max(im);
    let smaller = re.min(im);
    let ratio = if larger == 0.0 { 0.0 } else { smaller / larger };
    (ratio * ratio + 1.0).sqrt() * larger
}

/// Hyperbolic tangent as NumPy computes it for `float32`: a degree-6
/// polynomial in `|x| - b` per interval of `|x|`, with the sign of `x` copied
/// onto the result.
pub(crate) fn tanh(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    // The exponent and top mantissa bits pick one of 32 intervals.
    let bits = (x.to_bits() & 0x7fe0_0000) as i32;
    let r = if bits <= 0x7f00_0000 {
        let idx = ((bits - 0x3d40_0000).clamp(0, 0x03e0_0000) >> 21) as usize;
        let c = |row: usize| f32::from_bits(TANH_TABLE[row][idx]);
        let y = x.abs() - c(0);
        let mut r = c(7) * y + c(6);
        for row in (1..6).rev() {
            r = r * y + c(row);
        }
        r
    } else {
        1.0
    };
    r.copysign(x)
}

/// Rows: interval start `b`, then the coefficients `c0` to `c6`.
const TANH_TABLE: [[u32; 32]; 8] = [
    // b
    [
        0x00000000, 0x3d700000, 0x3d900000, 0x3db00000, 0x3dd00000, 0x3df00000, 0x3e100000,
        0x3e300000, 0x3e500000, 0x3e700000, 0x3e900000, 0x3eb00000, 0x3ed00000, 0x3ef00000,
        0x3f100000, 0x3f300000, 0x3f500000, 0x3f700000, 0x3f900000, 0x3fb00000, 0x3fd00000,
        0x3ff00000, 0x40100000, 0x40300000, 0x40500000, 0x40700000, 0x40900000, 0x40b00000,
        0x40d00000, 0x40f00000, 0x41100000, 0x00000000,
    ],
    // c0
    [
        0x00000000, 0x3d6fb9c9, 0x3d8fc35f, 0x3daf9169, 0x3dcf49ab, 0x3deee849, 0x3e0f0ee8,
        0x3e2e4984, 0x3e4d2f8e, 0x3e6bb32e, 0x3e8c51cd, 0x3ea96163, 0x3ec543f1, 0x3edfd735,
        0x3f028438, 0x3f18abf0, 0x3f2bc480, 0x3f3bec1c, 0x3f4f2e5b, 0x3f613c53, 0x3f6ce37d,
        0x3f743c4f, 0x3f7a5feb, 0x3f7dea85, 0x3f7f3b3d, 0x3f7fb78c, 0x3f7fefd4, 0x3f7ffdd0,
        0x3f7fffb4, 0x3f7ffff6, 0x3f7fffff, 0x3f800000,
    ],
    // c1
    [
        0x3f800000, 0x3f7f1f84, 0x3f7ebd11, 0x3f7e1e5f, 0x3f7d609f, 0x3f7c842d, 0x3f7b00e5,
        0x3f789580, 0x3f75b8ad, 0x3f726fd9, 0x3f6cc59b, 0x3f63fb92, 0x3f59ff97, 0x3f4f11d7,
        0x3f3d7573, 0x3f24f360, 0x3f0cbfe7, 0x3eec1a69, 0x3eb0a801, 0x3e6753a2, 0x3e132f1a,
        0x3db7e7d3, 0x3d320845, 0x3c84d3d4, 0x3bc477b7, 0x3b10d3da, 0x3a01601e, 0x388c1a3b,
        0x3717b0da, 0x35a43bce, 0x338306c6, 0x00000000,
    ],
    // c2
    [
        0xb0343c7b, 0xbd6ee69d, 0xbd8f0da7, 0xbdae477d, 0xbdcd2a1f, 0xbdeba80d, 0xbe0c443b,
        0xbe293cf3, 0xbe44f282, 0xbe5f3651, 0xbe81c7c0, 0xbe96d7ca, 0xbea7fb8e, 0xbeb50e9e,
        0xbec12efe, 0xbec4be92, 0xbebce070, 0xbead510e, 0xbe8ef7d6, 0xbe4b8704, 0xbe083237,
        0xbdaf7449, 0xbd2e1ec4, 0xbc83bf06, 0xbbc3e0b5, 0xbb10aadc, 0xba0157db, 0xb88c18f2,
        0xb717b096, 0xb5a43bae, 0xb383012c, 0x00000000,
    ],
    // c3
    [
        0xbeaaaaa5, 0xbeab0612, 0xbea7f01f, 0xbea4e120, 0xbea387b7, 0xbea15962, 0xbe9d57f7,
        0xbe976b5a, 0xbe90230d, 0xbe880dff, 0xbe7479b3, 0xbe4c3d88, 0xbe212482, 0xbdeb8cba,
        0xbd5e78ad, 0x3c6b5e6e, 0x3d839143, 0x3dc21ee1, 0x3de347af, 0x3dcbec96, 0x3d99ef2d,
        0x3d542ea1, 0x3cdde701, 0x3c2cca67, 0x3b81cb27, 0x3ac073a1, 0x39ac3032, 0x383a94d9,
        0x36ca081d, 0x355abd4c, 0x332b3cb6, 0x00000000,
    ],
    // c4
    [
        0xb76dd6b9, 0xbe1c276d, 0x3c1dcf2f, 0x3dc1a78d, 0x3d96f985, 0x3da2b61b, 0x3dc13397,
        0x3dd2f670, 0x3df48a0a, 0x3e06c5a8, 0x3e1a3aba, 0x3e27c405, 0x3e2e78d0, 0x3e2c3e44,
        0x3e1d3097, 0x3df4a8f4, 0x3da38508, 0x3d31416a, 0x3b562657, 0xbcaeeac9, 0xbcce9419,
        0xbcaaeac4, 0xbc49e7d0, 0xbba71ddd, 0xbb003b0e, 0xba3f9a05, 0xb92c08a7, 0xb7ba9232,
        0xb64a0b0f, 0xb4dac169, 0xb2ab78ac, 0x00000000,
    ],
    // c5
    [
        0x3e0910e9, 0x43761143, 0x4165ecdc, 0xc190f756, 0xc08c097d, 0xc02ba813, 0xbf7f6bda,
        0x3f2b1dc0, 0x3ece105d, 0x3f426a94, 0xbadb0dc4, 0x3da43b17, 0xbd51ab88, 0xbcaea23d,
        0xbd3b6d8d, 0xbd6caaad, 0xbd795bed, 0xbd5fddda, 0xbd038f3b, 0xbc1cad63, 0x3abb4766,
        0x3b95f10b, 0x3b825873, 0x3afaea66, 0x3a49f878, 0x39996bf3, 0x388f3e6c, 0x371bb0e3,
        0x35a8a5e6, 0x34369b17, 0x322487b0, 0x00000000,
    ],
    // c6
    [
        0xbc0e2f66, 0x460bda12, 0x43d638ef, 0xc3e11c3e, 0xc2baa4e9, 0xc249da2d, 0xc1859b82,
        0x40dd5b57, 0x40494640, 0x40c730a8, 0xbf0f160e, 0x3e30e76f, 0xbea81387, 0xbdb26a1c,
        0xbd351e57, 0xbb4c01a0, 0x3c1d7bfb, 0x3c722cd1, 0x3c973f1c, 0x3c33a31b, 0x3b862ef4,
        0x3a27b3d0, 0xba3b5907, 0xba0efc22, 0xb97f9f0f, 0xb8c8af50, 0xb7bdddfb, 0xb64f2950,
        0xb4e085b1, 0xb3731dfa, 0xb15a1f04, 0x00000000,
    ],
];

#[cfg(test)]
mod tests {
    use super::{complex_abs, log10, pairwise_sum, tanh};

    #[test]
    fn tanh_handles_edge_values() {
        assert_eq!(tanh(-0.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(tanh(100.0), 1.0);
        assert_eq!(tanh(f32::NEG_INFINITY), -1.0);
        assert!(tanh(f32::NAN).is_nan());
    }

    #[test]
    fn tanh_is_within_two_ulps_of_libm() {
        for i in -2000..2000 {
            let x = i as f32 * 0.0057;
            let (a, b) = (tanh(x).to_bits() as i64, libm::tanhf(x).to_bits() as i64);
            assert!((a - b).abs() <= 2, "tanh({x})");
        }
    }

    #[test]
    fn log10_matches_reference_platform() {
        // np.log10(np.float32(x)) on macOS arm64: a plain input, then one
        // from each table.
        for (x, bits) in [
            (2.5f32, 0x3ecbbecbu32),
            (f32::from_bits(0x3f8070d8), 0x3ac3b163),
        ] {
            assert_eq!(log10(x).to_bits(), bits, "log10({x})");
        }
        let down = f32::from_bits(super::LOG10_DOWN[0]);
        assert_eq!(
            log10(down).to_bits() + 1,
            (libm::log10(down as f64) as f32).to_bits()
        );
        assert!(super::LOG10_UP.windows(2).all(|w| w[0] < w[1]));
        assert!(super::LOG10_DOWN.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn complex_abs_handles_edge_values() {
        assert_eq!(complex_abs(3.0, -4.0), 5.0);
        assert_eq!(complex_abs(0.0, 0.0), 0.0);
        assert_eq!(complex_abs(f32::INFINITY, f32::NAN), f32::INFINITY);
    }

    #[test]
    fn pairwise_sum_matches_numpy() {
        let values: Vec<f32> = (0..300)
            .map(|i| ((i * 7919) % 1000) as f32 / 997.0)
            .collect();
        // np.sum(values[:n]) for float32
        for (n, bits) in [(5, 0x404cc63au32), (37, 0x4190ddd9), (300, 0x43169a10)] {
            assert_eq!(pairwise_sum(&values[..n]).to_bits(), bits, "n = {n}");
        }
    }
}
