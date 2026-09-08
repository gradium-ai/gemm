#![cfg_attr(not(feature = "std"), no_std)]
#![warn(rust_2018_idioms)]

mod gemm;

#[cfg(feature = "f16")]
pub use crate::gemm::f16;
pub use crate::gemm::{c32, c64, gemm};
pub use gemm_common::packed_cache;
pub use gemm_common::Parallelism;

pub use gemm_common::gemm::{
    get_lhs_packing_threshold_multi_thread, get_lhs_packing_threshold_single_thread,
    get_rhs_packing_threshold, get_threading_threshold, set_lhs_packing_threshold_multi_thread,
    set_lhs_packing_threshold_single_thread, set_rhs_packing_threshold, set_threading_threshold,
    DEFAULT_LHS_PACKING_THRESHOLD_MULTI_THREAD, DEFAULT_LHS_PACKING_THRESHOLD_SINGLE_THREAD,
    DEFAULT_RHS_PACKING_THRESHOLD, DEFAULT_THREADING_THRESHOLD,
};
pub use gemm_common::{get_wasm_simd128, set_wasm_simd128, DEFAULT_WASM_SIMD128};

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::{vec, vec::Vec};
    use num_traits::Float;

    /// `dst = lhs * rhs`, with a column-major dst and a row-major rhs.
    fn matmul_f32(
        m: usize,
        n: usize,
        k: usize,
        (lhs_rs, lhs_cs): (isize, isize),
        lhs: &[f32],
        rhs: &[f32],
        parallelism: Parallelism,
    ) -> Vec<f32> {
        let mut dst = vec![0f32; m * n];
        // SAFETY: shapes and strides match the buffers, which are all sized to suit.
        unsafe {
            gemm(
                m, n, k, dst.as_mut_ptr(), m as isize, 1, false, lhs.as_ptr(), lhs_cs, lhs_rs,
                rhs.as_ptr(), k as isize, 1, 0f32, 1f32, false, false, false, parallelism,
            );
        }
        dst
    }

    /// A cached lhs panel must give the same answer as packing per call. One test, since
    /// enabling the cache is a process-wide switch.
    #[test]
    fn test_packed_lhs_cache_hit_matches_miss() {
        // (m, n, k, row_major_lhs). The row-major cases cover the widened path: uncached,
        // a row-contiguous lhs is read in place rather than packed.
        let cases = [
            (64, 16, 128, false),
            (96, 33, 100, false), // m not a multiple of MR, ragged n
            (37, 5, 71, false),
            (512, 16, 384, true),
            (1536, 16, 512, true), // m > 2 * mc, so never prepacked before
            (512, 128, 300, true), // several column chunks
        ];

        for (m, n, k, row_major) in cases {
            let strides = if row_major { (k as isize, 1) } else { (1, m as isize) };
            let lhs: Vec<f32> =
                (0..m * k).map(|i| ((i * 37 % 101) as f32) / 101.0 - 0.5).collect();
            let rhs: Vec<f32> =
                (0..k * n).map(|i| ((i * 53 % 97) as f32) / 97.0 - 0.5).collect();

            packed_cache::set_enabled(false);
            let want = matmul_f32(m, n, k, strides, &lhs, &rhs, Parallelism::None);

            packed_cache::set_enabled(true);
            packed_cache::set_budget_mb(1024);
            packed_cache::clear();
            // Three rounds: the probe, the call that packs, then the calls that hit.
            for round in 0..3 {
                for parallelism in [Parallelism::None, Parallelism::Rayon(0)] {
                    let got = matmul_f32(m, n, k, strides, &lhs, &rhs, parallelism);
                    let worst = got
                        .iter()
                        .zip(&want)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    assert!(
                        worst < 1e-4,
                        "{m}x{k} * {k}x{n} row_major={row_major} round={round} \
                         {parallelism:?}: max error {worst}"
                    );
                }
            }
            assert!(packed_cache::stats().1 > 0, "the cached path was never taken");
        }

        packed_cache::clear();
        packed_cache::set_enabled(false);
    }

    #[test]
    fn test_gemm_f16() {
        let mut mnks = vec![];
        mnks.push((4, 4, 4));
        mnks.push((63, 2, 10));
        mnks.push((16, 2, 1));
        mnks.push((0, 0, 4));
        mnks.push((16, 1, 1));
        mnks.push((16, 3, 1));
        mnks.push((16, 4, 1));
        mnks.push((16, 1, 2));
        mnks.push((16, 2, 2));
        mnks.push((16, 3, 2));
        mnks.push((16, 4, 2));
        mnks.push((16, 16, 1));
        mnks.push((64, 64, 0));
        mnks.push((256, 256, 256));
        mnks.push((4096, 4096, 4));
        mnks.push((64, 64, 4));
        mnks.push((0, 64, 4));
        mnks.push((64, 0, 4));
        mnks.push((8, 16, 1));
        mnks.push((16, 8, 1));
        mnks.push((1, 1, 2));
        mnks.push((1024, 1024, 1));
        mnks.push((1024, 1024, 4));
        mnks.push((63, 1, 10));
        mnks.push((63, 3, 10));
        mnks.push((63, 4, 10));
        mnks.push((1, 63, 10));
        mnks.push((2, 63, 10));
        mnks.push((3, 63, 10));
        mnks.push((4, 63, 10));

        // gemv shapes above the threading threshold. k is kept modest because the
        // f16 fallback accumulates in f16.
        mnks.push((1, 2048, 384));
        mnks.push((2048, 1, 384));
        for (m, n, k) in mnks {
            #[cfg(feature = "std")]
            dbg!(m, n, k);
            for parallelism in [
                Parallelism::None,
                #[cfg(feature = "rayon")]
                Parallelism::Rayon(0),
                #[cfg(feature = "rayon")]
                Parallelism::Rayon(128),
            ] {
                for alpha in [0.0, 1.0, 2.3] {
                    for beta in [0.0, 1.0, 2.3] {
                        #[cfg(feature = "std")]
                        dbg!(alpha, beta, parallelism);

                        for colmajor in [true, false] {
                            let alpha = f16::from_f32(alpha);
                            let beta = f16::from_f32(beta);
                            let a_vec: Vec<f16> = (0..(m * k))
                                .map(|_| f16::from_f32(rand::random()))
                                .collect();
                            let b_vec: Vec<f16> = (0..(k * n))
                                .map(|_| f16::from_f32(rand::random()))
                                .collect();
                            let mut c_vec: Vec<f16> = (0..(m * n))
                                .map(|_| f16::from_f32(rand::random()))
                                .collect();
                            let mut d_vec = c_vec.clone();

                            unsafe {
                                gemm::gemm(
                                    m,
                                    n,
                                    k,
                                    c_vec.as_mut_ptr(),
                                    if colmajor { m } else { 1 } as isize,
                                    if colmajor { 1 } else { n } as isize,
                                    true,
                                    a_vec.as_ptr(),
                                    m as isize,
                                    1,
                                    b_vec.as_ptr(),
                                    k as isize,
                                    1,
                                    alpha,
                                    beta,
                                    false,
                                    false,
                                    false,
                                    parallelism,
                                );

                                gemm::gemm_fallback(
                                    m,
                                    n,
                                    k,
                                    d_vec.as_mut_ptr(),
                                    if colmajor { m } else { 1 } as isize,
                                    if colmajor { 1 } else { n } as isize,
                                    true,
                                    a_vec.as_ptr(),
                                    m as isize,
                                    1,
                                    b_vec.as_ptr(),
                                    k as isize,
                                    1,
                                    alpha,
                                    beta,
                                );
                            }
                            let eps = f16::from_f32(1e-1);
                            for (c, d) in c_vec.iter().zip(d_vec.iter()) {
                                let eps_rel = c.abs() * eps;
                                let eps_abs = eps;
                                let eps = if eps_rel > eps_abs { eps_rel } else { eps_abs };
                                assert_approx_eq::assert_approx_eq!(c, d, eps);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_gemm_f32() {
        set_wasm_simd128(true);

        let mut mnks = vec![];
        mnks.push((63, 2, 10));
        mnks.push((1, 2, 10));
        mnks.push((1, 63, 10));

        // large m to trigger parallelized rhs packing with big number of threads and small n
        mnks.push((2048, 255, 255));

        mnks.push((256, 256, 256));
        mnks.push((4096, 4096, 4));
        mnks.push((64, 64, 4));
        mnks.push((0, 64, 4));
        mnks.push((64, 0, 4));
        mnks.push((0, 0, 4));
        mnks.push((64, 64, 0));
        mnks.push((16, 1, 1));
        mnks.push((16, 2, 1));
        mnks.push((16, 3, 1));
        mnks.push((16, 4, 1));
        mnks.push((16, 1, 2));
        mnks.push((16, 2, 2));
        mnks.push((16, 3, 2));
        mnks.push((16, 4, 2));
        mnks.push((16, 16, 1));
        mnks.push((8, 16, 1));
        mnks.push((16, 8, 1));
        mnks.push((1, 1, 2));
        mnks.push((4, 4, 4));
        mnks.push((1024, 1024, 1));
        mnks.push((1024, 1024, 4));
        mnks.push((63, 1, 10));
        mnks.push((63, 3, 10));
        mnks.push((63, 4, 10));
        mnks.push((2, 63, 10));
        mnks.push((3, 63, 10));
        mnks.push((4, 63, 10));

        // gemv shapes above the threading threshold. depth is kept short so the f32
        // accumulation stays within the fixed absolute tolerance used below.
        mnks.push((1, 8192, 128));
        mnks.push((8192, 1, 128));
        for (m, n, k) in mnks {
            #[cfg(feature = "std")]
            dbg!(m, n, k);
            for parallelism in [
                Parallelism::None,
                #[cfg(feature = "rayon")]
                Parallelism::Rayon(0),
                #[cfg(feature = "rayon")]
                Parallelism::Rayon(128),
            ] {
                for alpha in [0.0, 1.0, 2.3] {
                    for beta in [0.0, 1.0, 2.3] {
                        #[cfg(feature = "std")]
                        dbg!(alpha, beta, parallelism);
                        for colmajor in [true, false] {
                            let a_vec: Vec<f32> = (0..(m * k)).map(|_| rand::random()).collect();
                            let b_vec: Vec<f32> = (0..(k * n)).map(|_| rand::random()).collect();
                            let mut c_vec: Vec<f32> =
                                (0..(m * n)).map(|_| rand::random()).collect();
                            let mut d_vec = c_vec.clone();

                            unsafe {
                                gemm::gemm(
                                    m,
                                    n,
                                    k,
                                    c_vec.as_mut_ptr(),
                                    if colmajor { m } else { 1 } as isize,
                                    if colmajor { 1 } else { n } as isize,
                                    true,
                                    a_vec.as_ptr(),
                                    m as isize,
                                    1,
                                    b_vec.as_ptr(),
                                    k as isize,
                                    1,
                                    alpha,
                                    beta,
                                    false,
                                    false,
                                    false,
                                    parallelism,
                                );

                                gemm::gemm_fallback(
                                    m,
                                    n,
                                    k,
                                    d_vec.as_mut_ptr(),
                                    if colmajor { m } else { 1 } as isize,
                                    if colmajor { 1 } else { n } as isize,
                                    true,
                                    a_vec.as_ptr(),
                                    m as isize,
                                    1,
                                    b_vec.as_ptr(),
                                    k as isize,
                                    1,
                                    alpha,
                                    beta,
                                );
                            }
                            for (c, d) in c_vec.iter().zip(d_vec.iter()) {
                                assert_approx_eq::assert_approx_eq!(c, d, 1e-3);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_gemm_f64() {
        set_wasm_simd128(true);

        let mut mnks = vec![];
        mnks.push((63, 2, 10));
        mnks.push((1, 2, 10));
        mnks.push((1, 63, 10));

        // large m to trigger parallelized rhs packing with big number of threads and small n
        mnks.push((2048, 255, 255));

        mnks.push((256, 256, 256));
        mnks.push((4096, 4096, 4));
        mnks.push((64, 64, 4));
        mnks.push((0, 64, 4));
        mnks.push((64, 0, 4));
        mnks.push((0, 0, 4));
        mnks.push((64, 64, 0));
        mnks.push((16, 1, 1));
        mnks.push((16, 2, 1));
        mnks.push((16, 3, 1));
        mnks.push((16, 4, 1));
        mnks.push((16, 1, 2));
        mnks.push((16, 2, 2));
        mnks.push((16, 3, 2));
        mnks.push((16, 4, 2));
        mnks.push((16, 16, 1));
        mnks.push((8, 16, 1));
        mnks.push((16, 8, 1));
        mnks.push((1, 1, 2));
        mnks.push((4, 4, 4));
        mnks.push((1024, 1024, 1));
        mnks.push((1024, 1024, 4));
        mnks.push((63, 1, 10));
        mnks.push((63, 3, 10));
        mnks.push((63, 4, 10));
        mnks.push((2, 63, 10));
        mnks.push((3, 63, 10));
        mnks.push((4, 63, 10));

        // gemv shapes above the threading threshold: the first two split the output
        // dimension, the last two are short-output/long-depth and take the k-split.
        mnks.push((1, 1024, 1024));
        mnks.push((1024, 1, 1024));
        mnks.push((1, 128, 8192));
        mnks.push((128, 1, 8192));
        for (m, n, k) in mnks {
            #[cfg(feature = "std")]
            dbg!(m, n, k);
            for parallelism in [
                Parallelism::None,
                #[cfg(feature = "rayon")]
                Parallelism::Rayon(0),
                #[cfg(feature = "rayon")]
                Parallelism::Rayon(128),
            ] {
                for alpha in [0.0, 1.0, 2.3] {
                    for beta in [0.0, 1.0, 2.3] {
                        #[cfg(feature = "std")]
                        dbg!(alpha, beta, parallelism);
                        for colmajor in [true, false] {
                            let a_vec: Vec<f64> = (0..(m * k)).map(|_| rand::random()).collect();
                            let b_vec: Vec<f64> = (0..(k * n)).map(|_| rand::random()).collect();
                            let mut c_vec: Vec<f64> =
                                (0..(m * n)).map(|_| rand::random()).collect();
                            let mut d_vec = c_vec.clone();

                            unsafe {
                                gemm::gemm(
                                    m,
                                    n,
                                    k,
                                    c_vec.as_mut_ptr(),
                                    if colmajor { m } else { 1 } as isize,
                                    if colmajor { 1 } else { n } as isize,
                                    true,
                                    a_vec.as_ptr(),
                                    m as isize,
                                    1,
                                    b_vec.as_ptr(),
                                    k as isize,
                                    1,
                                    alpha,
                                    beta,
                                    false,
                                    false,
                                    false,
                                    parallelism,
                                );

                                gemm::gemm_fallback(
                                    m,
                                    n,
                                    k,
                                    d_vec.as_mut_ptr(),
                                    if colmajor { m } else { 1 } as isize,
                                    if colmajor { 1 } else { n } as isize,
                                    true,
                                    a_vec.as_ptr(),
                                    m as isize,
                                    1,
                                    b_vec.as_ptr(),
                                    k as isize,
                                    1,
                                    alpha,
                                    beta,
                                );
                            }
                            for (c, d) in c_vec.iter().zip(d_vec.iter()) {
                                assert_approx_eq::assert_approx_eq!(c, d);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_gemm_cplx32() {
        let mut mnks = vec![];
        mnks.push((4, 4, 4));
        mnks.push((0, 64, 4));
        mnks.push((64, 0, 4));
        mnks.push((0, 0, 4));
        mnks.push((64, 64, 4));
        mnks.push((64, 64, 0));
        mnks.push((6, 3, 1));
        mnks.push((1, 1, 2));
        mnks.push((128, 128, 128));
        mnks.push((16, 1, 1));
        mnks.push((16, 2, 1));
        mnks.push((16, 3, 1));
        mnks.push((16, 4, 1));
        mnks.push((16, 1, 2));
        mnks.push((16, 2, 2));
        mnks.push((16, 3, 2));
        mnks.push((16, 4, 2));
        mnks.push((16, 16, 1));
        mnks.push((8, 16, 1));
        mnks.push((16, 8, 1));
        mnks.push((1024, 1024, 4));
        mnks.push((1024, 1024, 1));
        mnks.push((63, 1, 10));
        mnks.push((63, 2, 10));
        mnks.push((63, 3, 10));
        mnks.push((63, 4, 10));
        mnks.push((1, 63, 10));
        mnks.push((2, 63, 10));
        mnks.push((3, 63, 10));
        mnks.push((4, 63, 10));

        // gemv shapes above the threading threshold. depth is kept short so the f32
        // accumulation stays within the fixed absolute tolerance used below.
        mnks.push((1, 8192, 128));
        mnks.push((8192, 1, 128));
        for (m, n, k) in mnks {
            #[cfg(feature = "std")]
            dbg!(m, n, k);

            let zero = c32::new(0.0, 0.0);
            let one = c32::new(1.0, 0.0);
            let arbitrary = c32::new(2.3, 4.1);
            for alpha in [zero, one, arbitrary] {
                for beta in [zero, one, arbitrary] {
                    #[cfg(feature = "std")]
                    dbg!(alpha, beta);
                    for conj_dst in [false, true] {
                        for conj_lhs in [false, true] {
                            for conj_rhs in [false, true] {
                                #[cfg(feature = "std")]
                                dbg!(conj_dst);
                                #[cfg(feature = "std")]
                                dbg!(conj_lhs);
                                #[cfg(feature = "std")]
                                dbg!(conj_rhs);
                                for colmajor in [true, false] {
                                    let a_vec: Vec<f32> =
                                        (0..(2 * m * k)).map(|_| rand::random()).collect();
                                    let b_vec: Vec<f32> =
                                        (0..(2 * k * n)).map(|_| rand::random()).collect();
                                    let mut c_vec: Vec<f32> =
                                        (0..(2 * m * n)).map(|_| rand::random()).collect();
                                    let mut d_vec = c_vec.clone();

                                    unsafe {
                                        gemm::gemm(
                                            m,
                                            n,
                                            k,
                                            c_vec.as_mut_ptr() as *mut c32,
                                            if colmajor { m } else { 1 } as isize,
                                            if colmajor { 1 } else { n } as isize,
                                            true,
                                            a_vec.as_ptr() as *const c32,
                                            m as isize,
                                            1,
                                            b_vec.as_ptr() as *const c32,
                                            k as isize,
                                            1,
                                            alpha,
                                            beta,
                                            conj_dst,
                                            conj_lhs,
                                            conj_rhs,
                                            #[cfg(feature = "rayon")]
                                            Parallelism::Rayon(0),
                                            #[cfg(not(feature = "rayon"))]
                                            Parallelism::None,
                                        );

                                        gemm::gemm_cplx_fallback(
                                            m,
                                            n,
                                            k,
                                            d_vec.as_mut_ptr() as *mut c32,
                                            if colmajor { m } else { 1 } as isize,
                                            if colmajor { 1 } else { n } as isize,
                                            true,
                                            a_vec.as_ptr() as *const c32,
                                            m as isize,
                                            1,
                                            b_vec.as_ptr() as *const c32,
                                            k as isize,
                                            1,
                                            alpha,
                                            beta,
                                            conj_dst,
                                            conj_lhs,
                                            conj_rhs,
                                        );
                                    }
                                    for (c, d) in c_vec.iter().zip(d_vec.iter()) {
                                        assert_approx_eq::assert_approx_eq!(c, d, 1e-3);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_gemm_cplx64() {
        let mut mnks = vec![];
        mnks.push((4, 4, 4));
        mnks.push((0, 64, 4));
        mnks.push((64, 0, 4));
        mnks.push((0, 0, 4));
        mnks.push((64, 64, 4));
        mnks.push((64, 64, 0));
        mnks.push((6, 3, 1));
        mnks.push((1, 1, 2));
        mnks.push((128, 128, 128));
        mnks.push((16, 1, 1));
        mnks.push((16, 2, 1));
        mnks.push((16, 3, 1));
        mnks.push((16, 4, 1));
        mnks.push((16, 1, 2));
        mnks.push((16, 2, 2));
        mnks.push((16, 3, 2));
        mnks.push((16, 4, 2));
        mnks.push((16, 16, 1));
        mnks.push((8, 16, 1));
        mnks.push((16, 8, 1));
        mnks.push((1024, 1024, 4));
        mnks.push((1024, 1024, 1));
        mnks.push((63, 1, 10));
        mnks.push((63, 2, 10));
        mnks.push((63, 3, 10));
        mnks.push((63, 4, 10));
        mnks.push((1, 63, 10));
        mnks.push((2, 63, 10));
        mnks.push((3, 63, 10));
        mnks.push((4, 63, 10));

        // gemv shapes above the threading threshold: the first two split the output
        // dimension, the last two are short-output/long-depth and take the k-split.
        mnks.push((1, 1024, 1024));
        mnks.push((1024, 1, 1024));
        mnks.push((1, 128, 8192));
        mnks.push((128, 1, 8192));
        for (m, n, k) in mnks {
            #[cfg(feature = "std")]
            dbg!(m, n, k);

            let zero = c64::new(0.0, 0.0);
            let one = c64::new(1.0, 0.0);
            let arbitrary = c64::new(2.3, 4.1);
            for alpha in [zero, one, arbitrary] {
                for beta in [zero, one, arbitrary] {
                    #[cfg(feature = "std")]
                    dbg!(alpha, beta);
                    for conj_dst in [false, true] {
                        for conj_lhs in [false, true] {
                            for conj_rhs in [false, true] {
                                #[cfg(feature = "std")]
                                dbg!(conj_dst);
                                #[cfg(feature = "std")]
                                dbg!(conj_lhs);
                                #[cfg(feature = "std")]
                                dbg!(conj_rhs);
                                for colmajor in [true, false] {
                                    let a_vec: Vec<f64> =
                                        (0..(2 * m * k)).map(|_| rand::random()).collect();
                                    let b_vec: Vec<f64> =
                                        (0..(2 * k * n)).map(|_| rand::random()).collect();
                                    let mut c_vec: Vec<f64> =
                                        (0..(2 * m * n)).map(|_| rand::random()).collect();
                                    let mut d_vec = c_vec.clone();

                                    unsafe {
                                        gemm::gemm(
                                            m,
                                            n,
                                            k,
                                            c_vec.as_mut_ptr() as *mut c64,
                                            if colmajor { m } else { 1 } as isize,
                                            if colmajor { 1 } else { n } as isize,
                                            true,
                                            a_vec.as_ptr() as *const c64,
                                            m as isize,
                                            1,
                                            b_vec.as_ptr() as *const c64,
                                            k as isize,
                                            1,
                                            alpha,
                                            beta,
                                            conj_dst,
                                            conj_lhs,
                                            conj_rhs,
                                            #[cfg(feature = "rayon")]
                                            Parallelism::Rayon(0),
                                            #[cfg(not(feature = "rayon"))]
                                            Parallelism::None,
                                        );

                                        gemm::gemm_cplx_fallback(
                                            m,
                                            n,
                                            k,
                                            d_vec.as_mut_ptr() as *mut c64,
                                            if colmajor { m } else { 1 } as isize,
                                            if colmajor { 1 } else { n } as isize,
                                            true,
                                            a_vec.as_ptr() as *const c64,
                                            m as isize,
                                            1,
                                            b_vec.as_ptr() as *const c64,
                                            k as isize,
                                            1,
                                            alpha,
                                            beta,
                                            conj_dst,
                                            conj_lhs,
                                            conj_rhs,
                                        );
                                    }
                                    for (c, d) in c_vec.iter().zip(d_vec.iter()) {
                                        assert_approx_eq::assert_approx_eq!(c, d);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
