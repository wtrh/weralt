//! This module represents the functions about FFT (Fast Fourier Transform)
//! implemented by Mr. Ooura, and wrapper functions implemented by M. Morise.
//! We can use these wrapper functions as well as the FFTW functions.
//! Please see the FFTW web-page to show the usage of the wrapper functions.
//! - Ooura FFT
//!     - (Japanese) http://www.kurims.kyoto-u.ac.jp/~ooura/index-j.html
//!     - (English) http://www.kurims.kyoto-u.ac.jp/~ooura/index.html
//! - FFTW
//!     - (English) http://www.fftw.org/

// Commands for FFT (This is the same as FFTW)
pub const FFT_FORWARD: u8 = 1;
pub const FFT_BACKWARD: u8 = 2;
pub const FFT_ESTIMATE: u8 = 3;

// Complex number for FFT
#[allow(non_camel_case_types)]
pub type fft_complex = [f64; 2];

// Struct used for FFT
#[allow(non_camel_case_types)]
#[derive(Default)]
pub struct fft_plan<'a> {
    n: usize,
    sign: u8,
    _flags: u8,
    c_in: Option<&'a [fft_complex]>,
    r#in: Option<&'a [f64]>,
    c_out: Option<&'a mut [fft_complex]>,
    out: Option<&'a mut [f64]>,
    input: Vec<f64>,
    ip: Vec<usize>,
    w: Vec<f64>,
}

//---

fn BackwardFFT(p: &mut fft_plan) {
    let p_c_in = p.c_in.as_deref().unwrap();
    match p.c_out.as_deref_mut() {
        None => {
            // c2r
            let p_out = p.out.as_deref_mut().unwrap();
            p.input[0] = p_c_in[0][0];
            p.input[1] = p_c_in[(p.n / 2)][0];
            for i in 1..(p.n / 2) {
                p.input[i * 2] = p_c_in[i][0];
                p.input[i * 2 + 1] = -p_c_in[i][1];
            }
            rdft(p.n, -1, &mut p.input, &p.ip, &p.w);
            for i in 0..p.n {
                p_out[i] = p.input[i] * 2.0;
            }
        }
        Some(p_c_out) => {
            // c2c
            for i in 0..p.n {
                p.input[i * 2] = p_c_in[i][0];
                p.input[i * 2 + 1] = p_c_in[i][1];
            }
            cdft(p.n * 2, -1, &mut p.input, &p.ip, &p.w);
            for i in 0..p.n {
                p_c_out[i][0] = p.input[i * 2];
                p_c_out[i][1] = -p.input[i * 2 + 1];
            }
        }
    }
}

fn ForwardFFT(p: &mut fft_plan) {
    let p_c_out = p.c_out.as_deref_mut().unwrap();
    match p.c_in.as_deref() {
        None => {
            // r2c
            let p_in = p.r#in.as_deref().unwrap();
            for i in 0..p.n {
                p.input[i] = p_in[i];
            }
            rdft(p.n, 1, &mut p.input, &p.ip, &p.w);
            p_c_out[0][0] = p.input[0];
            p_c_out[0][1] = 0.0;
            for i in 1..(p.n / 2) {
                p_c_out[i][0] = p.input[i * 2];
                p_c_out[i][1] = -p.input[i * 2 + 1];
            }
            p_c_out[(p.n / 2)][0] = p.input[1];
            p_c_out[(p.n / 2)][1] = 0.0;
        }
        Some(p_c_in) => {
            // c2c
            for i in 0..p.n {
                p.input[i * 2] = p_c_in[i][0];
                p.input[i * 2 + 1] = p_c_in[i][1];
            }
            cdft(p.n * 2, 1, &mut p.input, &p.ip, &p.w);
            for i in 0..p.n {
                p_c_out[i][0] = p.input[i * 2];
                p_c_out[i][1] = -p.input[i * 2 + 1];
            }
        }
    }
}

//---

pub fn fft_plan_dft_1d<'a>(
    n: usize,
    r#in: Option<&'a [fft_complex]>,
    out: Option<&'a mut [fft_complex]>,
    sign: u8,
    _flags: u8,
) -> fft_plan<'a> {
    let mut output = fft_plan {
        n,
        r#in: None,
        c_in: r#in,
        out: None,
        c_out: out,
        sign,
        _flags,
        input: vec![0.0; n * 2],
        ip: vec![0; n],
        w: vec![0.0; n * 5 / 4],
    };
    output.ip[0] = 0;
    makewt(output.n >> 1, &mut output.ip, &mut output.w);
    return output;
}

pub fn fft_plan_dft_c2r_1d<'a>(
    n: usize,
    r#in: Option<&'a [fft_complex]>,
    out: Option<&'a mut [f64]>,
    _flags: u8,
) -> fft_plan<'a> {
    let mut output = fft_plan {
        n,
        r#in: None,
        c_in: r#in,
        out,
        c_out: None,
        sign: FFT_BACKWARD,
        _flags,
        input: vec![0.0; n],
        ip: vec![0; n],
        w: vec![0.0; n * 5 / 4],
    };

    output.ip[0] = 0;
    makewt(output.n >> 2, &mut output.ip, &mut output.w);
    makect(
        output.n >> 2,
        &mut output.ip,
        &mut output.w[(output.n >> 2)..],
    );
    return output;
}

pub fn fft_plan_dft_r2c_1d<'a>(
    n: usize,
    r#in: Option<&'a [f64]>,
    out: Option<&'a mut [fft_complex]>,
    _flags: u8,
) -> fft_plan<'a> {
    let mut output = fft_plan {
        n,
        r#in,
        c_in: None,
        out: None,
        c_out: out,
        sign: FFT_FORWARD,
        _flags,
        input: vec![0.0; n],
        ip: vec![0; n],
        w: vec![0.0; n * 5 / 4],
    };

    output.ip[0] = 0;
    makewt(output.n >> 2, &mut output.ip, &mut output.w);
    makect(
        output.n >> 2,
        &mut output.ip,
        &mut output.w[(output.n >> 2)..],
    );
    return output;
}

pub fn fft_execute(p: &mut fft_plan) {
    if p.sign == FFT_FORWARD {
        ForwardFFT(p);
    } else {
        // ifft
        BackwardFFT(p);
    }
}

//-----------------------------------------------------------------------
// The following functions are reffered by
// http://www.kurims.kyoto-u.ac.jp/~ooura/index.html

fn cdft(n: usize, isgn: i32, a: &mut [f64], ip: &[usize], w: &[f64]) {
    let nw = ip[0];
    if isgn >= 0 {
        cftfsub(n, a, ip, nw, w);
    } else {
        cftbsub(n, a, ip, nw, w);
    }
}

fn rdft(n: usize, isgn: i32, a: &mut [f64], ip: &[usize], w: &[f64]) {
    let nw = ip[0];
    let nc = ip[1];

    if isgn >= 0 {
        if n > 4 {
            cftfsub(n, a, ip, nw, w);
            rftfsub(n, a, nc, &w[nw..]);
        } else if n == 4 {
            cftfsub(n, a, ip, nw, w);
        }
        let xi = a[0] - a[1];
        a[0] += a[1];
        a[1] = xi;
    } else {
        a[1] = 0.5 * (a[0] - a[1]);
        a[0] -= a[1];
        if n > 4 {
            rftbsub(n, a, nc, &w[nw..]);
            cftbsub(n, a, ip, nw, w);
        } else if n == 4 {
            cftbsub(n, a, ip, nw, w);
        }
    }
}

fn makewt(nw: usize, ip: &mut [usize], w: &mut [f64]) {
    ip[0] = nw;
    ip[1] = 1;
    if nw > 2 {
        let mut nwh = nw >> 1;
        let delta: f64 = 1.0_f64.atan() / nwh as f64;
        let wn4r: f64 = (delta * nwh as f64).cos();
        w[0] = 1.0;
        w[1] = wn4r;
        if nwh == 4 {
            w[2] = (delta * 2.0).cos();
            w[3] = (delta * 2.0).sin();
        } else if nwh > 4 {
            makeipt(nw, ip);
            w[2] = 0.5 / (delta * 2.0).cos();
            w[3] = 0.5 / (delta * 6.0).cos();
            for j in (4..nwh).step_by(4) {
                w[j] = (delta * j as f64).cos();
                w[j + 1] = (delta * j as f64).sin();
                w[j + 2] = (3.0 * delta * j as f64).cos();
                w[j + 3] = -(3.0 * delta * j as f64).sin();
            }
        }
        let mut nw0 = 0;
        while nwh > 2 {
            let nw1 = nw0 + nwh;
            nwh >>= 1;
            w[nw1] = 1.0;
            w[nw1 + 1] = wn4r;
            if nwh == 4 {
                let wk1r = w[nw0 + 4];
                let wk1i = w[nw0 + 5];
                w[nw1 + 2] = wk1r;
                w[nw1 + 3] = wk1i;
            } else if nwh > 4 {
                let wk1r = w[nw0 + 4];
                let wk3r = w[nw0 + 6];
                w[nw1 + 2] = 0.5 / wk1r;
                w[nw1 + 3] = 0.5 / wk3r;
                for j in (4..nwh).step_by(4) {
                    let wk1r = w[nw0 + 2 * j];
                    let wk1i = w[nw0 + 2 * j + 1];
                    let wk3r = w[nw0 + 2 * j + 2];
                    let wk3i = w[nw0 + 2 * j + 3];
                    w[nw1 + j] = wk1r;
                    w[nw1 + j + 1] = wk1i;
                    w[nw1 + j + 2] = wk3r;
                    w[nw1 + j + 3] = wk3i;
                }
            }
            nw0 = nw1;
        }
    }
}

fn makeipt(nw: usize, ip: &mut [usize]) {
    ip[2] = 0;
    ip[3] = 16;
    let mut m = 2;
    let mut l = nw;
    while l > 32 {
        let m2 = m << 1;
        let q = m2 << 3;
        for j in m..m2 {
            let p = ip[j] << 2;
            ip[m + j] = p;
            ip[m2 + j] = p + q;
        }
        m = m2;
        l >>= 2;
    }
}

fn makect(nc: usize, ip: &mut [usize], c: &mut [f64]) {
    ip[1] = nc;
    if nc > 1 {
        let nch = nc >> 1;
        let delta: f64 = 1.0_f64.atan() / nch as f64;
        c[0] = (delta * nch as f64).cos();
        c[nch] = 0.5 * c[0];
        for j in 1..nch {
            c[j] = 0.5 * (delta * j as f64).cos();
            c[nc - j] = 0.5 * (delta * j as f64).sin();
        }
    }
}

// -------- child routines --------

fn cftfsub(n: usize, a: &mut [f64], ip: &[usize], nw: usize, w: &[f64]) {
    if n > 8 {
        if n > 32 {
            cftf1st(n, a, &w[(nw - (n >> 2))..]);
            if n > 512 {
                cftrec4(n, a, nw, w);
            } else if n > 128 {
                cftleaf(n, 1, a, nw, w);
            } else {
                cftfx41(n, a, nw, w);
            }
            bitrv2(n, ip, a);
        } else if n == 32 {
            cftf161(a, &w[nw - 8..]);
            bitrv216(a);
        } else {
            cftf081(a, w);
            bitrv208(a);
        }
    } else if n == 8 {
        cftf040(a);
    } else if n == 4 {
        cftx020(a);
    }
}

fn cftbsub(n: usize, a: &mut [f64], ip: &[usize], nw: usize, w: &[f64]) {
    if n > 8 {
        if n > 32 {
            cftb1st(n, a, &w[(nw - (n >> 2))..]);
            if n > 512 {
                cftrec4(n, a, nw, w);
            } else if n > 128 {
                cftleaf(n, 1, a, nw, w);
            } else {
                cftfx41(n, a, nw, w);
            }
            bitrv2conj(n, ip, a);
        } else if n == 32 {
            cftf161(a, &w[nw - 8..]);
            bitrv216neg(a);
        } else {
            cftf081(a, w);
            bitrv208neg(a);
        }
    } else if n == 8 {
        cftb040(a);
    } else if n == 4 {
        cftx020(a);
    }
}

fn bitrv2(n: usize, ip: &[usize], a: &mut [f64]) {
    let mut j1: usize;
    let mut k1: usize;
    let mut xr: f64;
    let mut xi: f64;
    let mut yr: f64;
    let mut yi: f64;

    let mut m = 1;
    let mut l = n >> 2;
    while l > 8 {
        m <<= 1;
        l >>= 2;
    }
    let nh = n >> 1;
    let nm = 4 * m;
    if l == 8 {
        for k in 0..m {
            for j in 0..k {
                j1 = 4 * j + 2 * ip[m + k];
                k1 = 4 * k + 2 * ip[m + j];
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 -= nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nh;
                k1 += 2;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 += nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += 2;
                k1 += nh;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 -= nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nh;
                k1 -= 2;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 += nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
            }
            k1 = 4 * k + 2 * ip[m + k];
            j1 = k1 + 2;
            k1 += nh;
            xr = a[j1];
            xi = a[j1 + 1];
            yr = a[k1];
            yi = a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 += nm;
            k1 += 2 * nm;
            xr = a[j1];
            xi = a[j1 + 1];
            yr = a[k1];
            yi = a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 += nm;
            k1 -= nm;
            xr = a[j1];
            xi = a[j1 + 1];
            yr = a[k1];
            yi = a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 -= 2;
            k1 -= nh;
            xr = a[j1];
            xi = a[j1 + 1];
            yr = a[k1];
            yi = a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 += nh + 2;
            k1 += nh + 2;
            xr = a[j1];
            xi = a[j1 + 1];
            yr = a[k1];
            yi = a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 -= nh - nm;
            k1 += 2 * nm - 2;
            xr = a[j1];
            xi = a[j1 + 1];
            yr = a[k1];
            yi = a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
        }
    } else {
        for k in 0..m {
            for j in 0..k {
                j1 = 4 * j + ip[m + k];
                k1 = 4 * k + ip[m + j];
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nh;
                k1 += 2;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += 2;
                k1 += nh;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nh;
                k1 -= 2;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
            }
            k1 = 4 * k + ip[m + k];
            j1 = k1 + 2;
            k1 += nh;
            xr = a[j1];
            xi = a[j1 + 1];
            yr = a[k1];
            yi = a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 += nm;
            k1 += nm;
            xr = a[j1];
            xi = a[j1 + 1];
            yr = a[k1];
            yi = a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
        }
    }
}

fn bitrv2conj(n: usize, ip: &[usize], a: &mut [f64]) {
    let mut j1: usize;
    let mut k1: usize;
    let mut xr: f64;
    let mut xi: f64;
    let mut yr: f64;
    let mut yi: f64;

    let mut m = 1;
    let mut l = n >> 2;
    while l > 8 {
        m <<= 1;
        l >>= 2;
    }
    let nh = n >> 1;
    let nm = 4 * m;
    if l == 8 {
        for k in 0..m {
            for j in 0..k {
                j1 = 4 * j + 2 * ip[m + k];
                k1 = 4 * k + 2 * ip[m + j];
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 -= nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nh;
                k1 += 2;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 += nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += 2;
                k1 += nh;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 -= nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nh;
                k1 -= 2;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 += nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
            }
            k1 = 4 * k + 2 * ip[m + k];
            j1 = k1 + 2;
            k1 += nh;
            a[j1 - 1] = -a[j1 - 1];
            xr = a[j1];
            xi = -a[j1 + 1];
            yr = a[k1];
            yi = -a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            a[k1 + 3] = -a[k1 + 3];
            j1 += nm;
            k1 += 2 * nm;
            xr = a[j1];
            xi = -a[j1 + 1];
            yr = a[k1];
            yi = -a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 += nm;
            k1 -= nm;
            xr = a[j1];
            xi = -a[j1 + 1];
            yr = a[k1];
            yi = -a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 -= 2;
            k1 -= nh;
            xr = a[j1];
            xi = -a[j1 + 1];
            yr = a[k1];
            yi = -a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 += nh + 2;
            k1 += nh + 2;
            xr = a[j1];
            xi = -a[j1 + 1];
            yr = a[k1];
            yi = -a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 -= nh - nm;
            k1 += 2 * nm - 2;
            a[j1 - 1] = -a[j1 - 1];
            xr = a[j1];
            xi = -a[j1 + 1];
            yr = a[k1];
            yi = -a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            a[k1 + 3] = -a[k1 + 3];
        }
    } else {
        for k in 0..m {
            for j in 0..k {
                j1 = 4 * j + ip[m + k];
                k1 = 4 * k + ip[m + j];
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nh;
                k1 += 2;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += 2;
                k1 += nh;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nh;
                k1 -= 2;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
            }
            k1 = 4 * k + ip[m + k];
            j1 = k1 + 2;
            k1 += nh;
            a[j1 - 1] = -a[j1 - 1];
            xr = a[j1];
            xi = -a[j1 + 1];
            yr = a[k1];
            yi = -a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            a[k1 + 3] = -a[k1 + 3];
            j1 += nm;
            k1 += nm;
            a[j1 - 1] = -a[j1 - 1];
            xr = a[j1];
            xi = -a[j1 + 1];
            yr = a[k1];
            yi = -a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            a[k1 + 3] = -a[k1 + 3];
        }
    }
}

fn bitrv216(a: &mut [f64]) {
    let x1r = a[2];
    let x1i = a[3];
    let x2r = a[4];
    let x2i = a[5];
    let x3r = a[6];
    let x3i = a[7];
    let x4r = a[8];
    let x4i = a[9];
    let x5r = a[10];
    let x5i = a[11];
    let x7r = a[14];
    let x7i = a[15];
    let x8r = a[16];
    let x8i = a[17];
    let x10r = a[20];
    let x10i = a[21];
    let x11r = a[22];
    let x11i = a[23];
    let x12r = a[24];
    let x12i = a[25];
    let x13r = a[26];
    let x13i = a[27];
    let x14r = a[28];
    let x14i = a[29];
    a[2] = x8r;
    a[3] = x8i;
    a[4] = x4r;
    a[5] = x4i;
    a[6] = x12r;
    a[7] = x12i;
    a[8] = x2r;
    a[9] = x2i;
    a[10] = x10r;
    a[11] = x10i;
    a[14] = x14r;
    a[15] = x14i;
    a[16] = x1r;
    a[17] = x1i;
    a[20] = x5r;
    a[21] = x5i;
    a[22] = x13r;
    a[23] = x13i;
    a[24] = x3r;
    a[25] = x3i;
    a[26] = x11r;
    a[27] = x11i;
    a[28] = x7r;
    a[29] = x7i;
}

fn bitrv216neg(a: &mut [f64]) {
    let x1r = a[2];
    let x1i = a[3];
    let x2r = a[4];
    let x2i = a[5];
    let x3r = a[6];
    let x3i = a[7];
    let x4r = a[8];
    let x4i = a[9];
    let x5r = a[10];
    let x5i = a[11];
    let x6r = a[12];
    let x6i = a[13];
    let x7r = a[14];
    let x7i = a[15];
    let x8r = a[16];
    let x8i = a[17];
    let x9r = a[18];
    let x9i = a[19];
    let x10r = a[20];
    let x10i = a[21];
    let x11r = a[22];
    let x11i = a[23];
    let x12r = a[24];
    let x12i = a[25];
    let x13r = a[26];
    let x13i = a[27];
    let x14r = a[28];
    let x14i = a[29];
    let x15r = a[30];
    let x15i = a[31];
    a[2] = x15r;
    a[3] = x15i;
    a[4] = x7r;
    a[5] = x7i;
    a[6] = x11r;
    a[7] = x11i;
    a[8] = x3r;
    a[9] = x3i;
    a[10] = x13r;
    a[11] = x13i;
    a[12] = x5r;
    a[13] = x5i;
    a[14] = x9r;
    a[15] = x9i;
    a[16] = x1r;
    a[17] = x1i;
    a[18] = x14r;
    a[19] = x14i;
    a[20] = x6r;
    a[21] = x6i;
    a[22] = x10r;
    a[23] = x10i;
    a[24] = x2r;
    a[25] = x2i;
    a[26] = x12r;
    a[27] = x12i;
    a[28] = x4r;
    a[29] = x4i;
    a[30] = x8r;
    a[31] = x8i;
}

fn bitrv208(a: &mut [f64]) {
    let x1r = a[2];
    let x1i = a[3];
    let x3r = a[6];
    let x3i = a[7];
    let x4r = a[8];
    let x4i = a[9];
    let x6r = a[12];
    let x6i = a[13];
    a[2] = x4r;
    a[3] = x4i;
    a[6] = x6r;
    a[7] = x6i;
    a[8] = x1r;
    a[9] = x1i;
    a[12] = x3r;
    a[13] = x3i;
}

fn bitrv208neg(a: &mut [f64]) {
    let x1r = a[2];
    let x1i = a[3];
    let x2r = a[4];
    let x2i = a[5];
    let x3r = a[6];
    let x3i = a[7];
    let x4r = a[8];
    let x4i = a[9];
    let x5r = a[10];
    let x5i = a[11];
    let x6r = a[12];
    let x6i = a[13];
    let x7r = a[14];
    let x7i = a[15];
    a[2] = x7r;
    a[3] = x7i;
    a[4] = x3r;
    a[5] = x3i;
    a[6] = x5r;
    a[7] = x5i;
    a[8] = x1r;
    a[9] = x1i;
    a[10] = x6r;
    a[11] = x6i;
    a[12] = x2r;
    a[13] = x2i;
    a[14] = x4r;
    a[15] = x4i;
}

fn cftf1st(n: usize, a: &mut [f64], w: &[f64]) {
    let mut wk1r;
    let mut wk1i;
    let mut wk3r;
    let mut wk3i;
    let mut j0;

    let mut y0r;
    let mut y0i;
    let mut y1r;
    let mut y1i;

    let mut y2r;
    let mut y2i;
    let mut y3r;
    let mut y3i;

    let mh = n >> 3;
    let m = 2 * mh;
    let mut j1 = m;
    let mut j2 = j1 + m;
    let mut j3 = j2 + m;
    let mut x0r = a[0] + a[j2];
    let mut x0i = a[1] + a[j2 + 1];
    let mut x1r = a[0] - a[j2];
    let mut x1i = a[1] - a[j2 + 1];
    let mut x2r = a[j1] + a[j3];
    let mut x2i = a[j1 + 1] + a[j3 + 1];
    let mut x3r = a[j1] - a[j3];
    let mut x3i = a[j1 + 1] - a[j3 + 1];
    a[0] = x0r + x2r;
    a[1] = x0i + x2i;
    a[j1] = x0r - x2r;
    a[j1 + 1] = x0i - x2i;
    a[j2] = x1r - x3i;
    a[j2 + 1] = x1i + x3r;
    a[j3] = x1r + x3i;
    a[j3 + 1] = x1i - x3r;
    let wn4r = w[1];
    let csc1 = w[2];
    let csc3 = w[3];
    let mut wd1r = 1.0;
    let mut wd1i = 0.0;
    let mut wd3r = 1.0;
    let mut wd3i = 0.0;
    let mut k = 0;
    for j in (2..mh - 2).step_by(4) {
        k += 4;
        wk1r = csc1 * (wd1r + w[k]);
        wk1i = csc1 * (wd1i + w[k + 1]);
        wk3r = csc3 * (wd3r + w[k + 2]);
        wk3i = csc3 * (wd3i + w[k + 3]);
        wd1r = w[k];
        wd1i = w[k + 1];
        wd3r = w[k + 2];
        wd3i = w[k + 3];
        j1 = j + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[j] + a[j2];
        x0i = a[j + 1] + a[j2 + 1];
        x1r = a[j] - a[j2];
        x1i = a[j + 1] - a[j2 + 1];
        y0r = a[j + 2] + a[j2 + 2];
        y0i = a[j + 3] + a[j2 + 3];
        y1r = a[j + 2] - a[j2 + 2];
        y1i = a[j + 3] - a[j2 + 3];
        x2r = a[j1] + a[j3];
        x2i = a[j1 + 1] + a[j3 + 1];
        x3r = a[j1] - a[j3];
        x3i = a[j1 + 1] - a[j3 + 1];
        y2r = a[j1 + 2] + a[j3 + 2];
        y2i = a[j1 + 3] + a[j3 + 3];
        y3r = a[j1 + 2] - a[j3 + 2];
        y3i = a[j1 + 3] - a[j3 + 3];
        a[j] = x0r + x2r;
        a[j + 1] = x0i + x2i;
        a[j + 2] = y0r + y2r;
        a[j + 3] = y0i + y2i;
        a[j1] = x0r - x2r;
        a[j1 + 1] = x0i - x2i;
        a[j1 + 2] = y0r - y2r;
        a[j1 + 3] = y0i - y2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a[j2] = wk1r * x0r - wk1i * x0i;
        a[j2 + 1] = wk1r * x0i + wk1i * x0r;
        x0r = y1r - y3i;
        x0i = y1i + y3r;
        a[j2 + 2] = wd1r * x0r - wd1i * x0i;
        a[j2 + 3] = wd1r * x0i + wd1i * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a[j3] = wk3r * x0r + wk3i * x0i;
        a[j3 + 1] = wk3r * x0i - wk3i * x0r;
        x0r = y1r + y3i;
        x0i = y1i - y3r;
        a[j3 + 2] = wd3r * x0r + wd3i * x0i;
        a[j3 + 3] = wd3r * x0i - wd3i * x0r;
        j0 = m - j;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[j0] + a[j2];
        x0i = a[j0 + 1] + a[j2 + 1];
        x1r = a[j0] - a[j2];
        x1i = a[j0 + 1] - a[j2 + 1];
        y0r = a[j0 - 2] + a[j2 - 2];
        y0i = a[j0 - 1] + a[j2 - 1];
        y1r = a[j0 - 2] - a[j2 - 2];
        y1i = a[j0 - 1] - a[j2 - 1];
        x2r = a[j1] + a[j3];
        x2i = a[j1 + 1] + a[j3 + 1];
        x3r = a[j1] - a[j3];
        x3i = a[j1 + 1] - a[j3 + 1];
        y2r = a[j1 - 2] + a[j3 - 2];
        y2i = a[j1 - 1] + a[j3 - 1];
        y3r = a[j1 - 2] - a[j3 - 2];
        y3i = a[j1 - 1] - a[j3 - 1];
        a[j0] = x0r + x2r;
        a[j0 + 1] = x0i + x2i;
        a[j0 - 2] = y0r + y2r;
        a[j0 - 1] = y0i + y2i;
        a[j1] = x0r - x2r;
        a[j1 + 1] = x0i - x2i;
        a[j1 - 2] = y0r - y2r;
        a[j1 - 1] = y0i - y2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a[j2] = wk1i * x0r - wk1r * x0i;
        a[j2 + 1] = wk1i * x0i + wk1r * x0r;
        x0r = y1r - y3i;
        x0i = y1i + y3r;
        a[j2 - 2] = wd1i * x0r - wd1r * x0i;
        a[j2 - 1] = wd1i * x0i + wd1r * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a[j3] = wk3i * x0r + wk3r * x0i;
        a[j3 + 1] = wk3i * x0i - wk3r * x0r;
        x0r = y1r + y3i;
        x0i = y1i - y3r;
        a[j3 - 2] = wd3i * x0r + wd3r * x0i;
        a[j3 - 1] = wd3i * x0i - wd3r * x0r;
    }
    wk1r = csc1 * (wd1r + wn4r);
    wk1i = csc1 * (wd1i + wn4r);
    wk3r = csc3 * (wd3r - wn4r);
    wk3i = csc3 * (wd3i - wn4r);
    j0 = mh;
    j1 = j0 + m;
    j2 = j1 + m;
    j3 = j2 + m;
    x0r = a[j0 - 2] + a[j2 - 2];
    x0i = a[j0 - 1] + a[j2 - 1];
    x1r = a[j0 - 2] - a[j2 - 2];
    x1i = a[j0 - 1] - a[j2 - 1];
    x2r = a[j1 - 2] + a[j3 - 2];
    x2i = a[j1 - 1] + a[j3 - 1];
    x3r = a[j1 - 2] - a[j3 - 2];
    x3i = a[j1 - 1] - a[j3 - 1];
    a[j0 - 2] = x0r + x2r;
    a[j0 - 1] = x0i + x2i;
    a[j1 - 2] = x0r - x2r;
    a[j1 - 1] = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    a[j2 - 2] = wk1r * x0r - wk1i * x0i;
    a[j2 - 1] = wk1r * x0i + wk1i * x0r;
    x0r = x1r + x3i;
    x0i = x1i - x3r;
    a[j3 - 2] = wk3r * x0r + wk3i * x0i;
    a[j3 - 1] = wk3r * x0i - wk3i * x0r;
    x0r = a[j0] + a[j2];
    x0i = a[j0 + 1] + a[j2 + 1];
    x1r = a[j0] - a[j2];
    x1i = a[j0 + 1] - a[j2 + 1];
    x2r = a[j1] + a[j3];
    x2i = a[j1 + 1] + a[j3 + 1];
    x3r = a[j1] - a[j3];
    x3i = a[j1 + 1] - a[j3 + 1];
    a[j0] = x0r + x2r;
    a[j0 + 1] = x0i + x2i;
    a[j1] = x0r - x2r;
    a[j1 + 1] = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    a[j2] = wn4r * (x0r - x0i);
    a[j2 + 1] = wn4r * (x0i + x0r);
    x0r = x1r + x3i;
    x0i = x1i - x3r;
    a[j3] = -wn4r * (x0r + x0i);
    a[j3 + 1] = -wn4r * (x0i - x0r);
    x0r = a[j0 + 2] + a[j2 + 2];
    x0i = a[j0 + 3] + a[j2 + 3];
    x1r = a[j0 + 2] - a[j2 + 2];
    x1i = a[j0 + 3] - a[j2 + 3];
    x2r = a[j1 + 2] + a[j3 + 2];
    x2i = a[j1 + 3] + a[j3 + 3];
    x3r = a[j1 + 2] - a[j3 + 2];
    x3i = a[j1 + 3] - a[j3 + 3];
    a[j0 + 2] = x0r + x2r;
    a[j0 + 3] = x0i + x2i;
    a[j1 + 2] = x0r - x2r;
    a[j1 + 3] = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    a[j2 + 2] = wk1i * x0r - wk1r * x0i;
    a[j2 + 3] = wk1i * x0i + wk1r * x0r;
    x0r = x1r + x3i;
    x0i = x1i - x3r;
    a[j3 + 2] = wk3i * x0r + wk3r * x0i;
    a[j3 + 3] = wk3i * x0i - wk3r * x0r;
}

fn cftb1st(n: usize, a: &mut [f64], w: &[f64]) {
    let mut wk1r;
    let mut wk1i;
    let mut wk3r;
    let mut wk3i;
    let mut j0;

    let mut y0r;
    let mut y0i;
    let mut y1r;
    let mut y1i;

    let mut y2r;
    let mut y2i;
    let mut y3r;
    let mut y3i;

    let mh = n >> 3;
    let m = 2 * mh;
    let mut j1 = m;
    let mut j2 = j1 + m;
    let mut j3 = j2 + m;
    let mut x0r = a[0] + a[j2];
    let mut x0i = -a[1] - a[j2 + 1];
    let mut x1r = a[0] - a[j2];
    let mut x1i = -a[1] + a[j2 + 1];
    let mut x2r = a[j1] + a[j3];
    let mut x2i = a[j1 + 1] + a[j3 + 1];
    let mut x3r = a[j1] - a[j3];
    let mut x3i = a[j1 + 1] - a[j3 + 1];
    a[0] = x0r + x2r;
    a[1] = x0i - x2i;
    a[j1] = x0r - x2r;
    a[j1 + 1] = x0i + x2i;
    a[j2] = x1r + x3i;
    a[j2 + 1] = x1i + x3r;
    a[j3] = x1r - x3i;
    a[j3 + 1] = x1i - x3r;
    let wn4r = w[1];
    let csc1 = w[2];
    let csc3 = w[3];
    let mut wd1r = 1.0;
    let mut wd1i = 0.0;
    let mut wd3r = 1.0;
    let mut wd3i = 0.0;
    let mut k = 0;
    for j in (2..mh - 2).step_by(4) {
        k += 4;
        wk1r = csc1 * (wd1r + w[k]);
        wk1i = csc1 * (wd1i + w[k + 1]);
        wk3r = csc3 * (wd3r + w[k + 2]);
        wk3i = csc3 * (wd3i + w[k + 3]);
        wd1r = w[k];
        wd1i = w[k + 1];
        wd3r = w[k + 2];
        wd3i = w[k + 3];
        j1 = j + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[j] + a[j2];
        x0i = -a[j + 1] - a[j2 + 1];
        x1r = a[j] - a[j2];
        x1i = -a[j + 1] + a[j2 + 1];
        y0r = a[j + 2] + a[j2 + 2];
        y0i = -a[j + 3] - a[j2 + 3];
        y1r = a[j + 2] - a[j2 + 2];
        y1i = -a[j + 3] + a[j2 + 3];
        x2r = a[j1] + a[j3];
        x2i = a[j1 + 1] + a[j3 + 1];
        x3r = a[j1] - a[j3];
        x3i = a[j1 + 1] - a[j3 + 1];
        y2r = a[j1 + 2] + a[j3 + 2];
        y2i = a[j1 + 3] + a[j3 + 3];
        y3r = a[j1 + 2] - a[j3 + 2];
        y3i = a[j1 + 3] - a[j3 + 3];
        a[j] = x0r + x2r;
        a[j + 1] = x0i - x2i;
        a[j + 2] = y0r + y2r;
        a[j + 3] = y0i - y2i;
        a[j1] = x0r - x2r;
        a[j1 + 1] = x0i + x2i;
        a[j1 + 2] = y0r - y2r;
        a[j1 + 3] = y0i + y2i;
        x0r = x1r + x3i;
        x0i = x1i + x3r;
        a[j2] = wk1r * x0r - wk1i * x0i;
        a[j2 + 1] = wk1r * x0i + wk1i * x0r;
        x0r = y1r + y3i;
        x0i = y1i + y3r;
        a[j2 + 2] = wd1r * x0r - wd1i * x0i;
        a[j2 + 3] = wd1r * x0i + wd1i * x0r;
        x0r = x1r - x3i;
        x0i = x1i - x3r;
        a[j3] = wk3r * x0r + wk3i * x0i;
        a[j3 + 1] = wk3r * x0i - wk3i * x0r;
        x0r = y1r - y3i;
        x0i = y1i - y3r;
        a[j3 + 2] = wd3r * x0r + wd3i * x0i;
        a[j3 + 3] = wd3r * x0i - wd3i * x0r;
        j0 = m - j;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[j0] + a[j2];
        x0i = -a[j0 + 1] - a[j2 + 1];
        x1r = a[j0] - a[j2];
        x1i = -a[j0 + 1] + a[j2 + 1];
        y0r = a[j0 - 2] + a[j2 - 2];
        y0i = -a[j0 - 1] - a[j2 - 1];
        y1r = a[j0 - 2] - a[j2 - 2];
        y1i = -a[j0 - 1] + a[j2 - 1];
        x2r = a[j1] + a[j3];
        x2i = a[j1 + 1] + a[j3 + 1];
        x3r = a[j1] - a[j3];
        x3i = a[j1 + 1] - a[j3 + 1];
        y2r = a[j1 - 2] + a[j3 - 2];
        y2i = a[j1 - 1] + a[j3 - 1];
        y3r = a[j1 - 2] - a[j3 - 2];
        y3i = a[j1 - 1] - a[j3 - 1];
        a[j0] = x0r + x2r;
        a[j0 + 1] = x0i - x2i;
        a[j0 - 2] = y0r + y2r;
        a[j0 - 1] = y0i - y2i;
        a[j1] = x0r - x2r;
        a[j1 + 1] = x0i + x2i;
        a[j1 - 2] = y0r - y2r;
        a[j1 - 1] = y0i + y2i;
        x0r = x1r + x3i;
        x0i = x1i + x3r;
        a[j2] = wk1i * x0r - wk1r * x0i;
        a[j2 + 1] = wk1i * x0i + wk1r * x0r;
        x0r = y1r + y3i;
        x0i = y1i + y3r;
        a[j2 - 2] = wd1i * x0r - wd1r * x0i;
        a[j2 - 1] = wd1i * x0i + wd1r * x0r;
        x0r = x1r - x3i;
        x0i = x1i - x3r;
        a[j3] = wk3i * x0r + wk3r * x0i;
        a[j3 + 1] = wk3i * x0i - wk3r * x0r;
        x0r = y1r - y3i;
        x0i = y1i - y3r;
        a[j3 - 2] = wd3i * x0r + wd3r * x0i;
        a[j3 - 1] = wd3i * x0i - wd3r * x0r;
    }
    wk1r = csc1 * (wd1r + wn4r);
    wk1i = csc1 * (wd1i + wn4r);
    wk3r = csc3 * (wd3r - wn4r);
    wk3i = csc3 * (wd3i - wn4r);
    j0 = mh;
    j1 = j0 + m;
    j2 = j1 + m;
    j3 = j2 + m;
    x0r = a[j0 - 2] + a[j2 - 2];
    x0i = -a[j0 - 1] - a[j2 - 1];
    x1r = a[j0 - 2] - a[j2 - 2];
    x1i = -a[j0 - 1] + a[j2 - 1];
    x2r = a[j1 - 2] + a[j3 - 2];
    x2i = a[j1 - 1] + a[j3 - 1];
    x3r = a[j1 - 2] - a[j3 - 2];
    x3i = a[j1 - 1] - a[j3 - 1];
    a[j0 - 2] = x0r + x2r;
    a[j0 - 1] = x0i - x2i;
    a[j1 - 2] = x0r - x2r;
    a[j1 - 1] = x0i + x2i;
    x0r = x1r + x3i;
    x0i = x1i + x3r;
    a[j2 - 2] = wk1r * x0r - wk1i * x0i;
    a[j2 - 1] = wk1r * x0i + wk1i * x0r;
    x0r = x1r - x3i;
    x0i = x1i - x3r;
    a[j3 - 2] = wk3r * x0r + wk3i * x0i;
    a[j3 - 1] = wk3r * x0i - wk3i * x0r;
    x0r = a[j0] + a[j2];
    x0i = -a[j0 + 1] - a[j2 + 1];
    x1r = a[j0] - a[j2];
    x1i = -a[j0 + 1] + a[j2 + 1];
    x2r = a[j1] + a[j3];
    x2i = a[j1 + 1] + a[j3 + 1];
    x3r = a[j1] - a[j3];
    x3i = a[j1 + 1] - a[j3 + 1];
    a[j0] = x0r + x2r;
    a[j0 + 1] = x0i - x2i;
    a[j1] = x0r - x2r;
    a[j1 + 1] = x0i + x2i;
    x0r = x1r + x3i;
    x0i = x1i + x3r;
    a[j2] = wn4r * (x0r - x0i);
    a[j2 + 1] = wn4r * (x0i + x0r);
    x0r = x1r - x3i;
    x0i = x1i - x3r;
    a[j3] = -wn4r * (x0r + x0i);
    a[j3 + 1] = -wn4r * (x0i - x0r);
    x0r = a[j0 + 2] + a[j2 + 2];
    x0i = -a[j0 + 3] - a[j2 + 3];
    x1r = a[j0 + 2] - a[j2 + 2];
    x1i = -a[j0 + 3] + a[j2 + 3];
    x2r = a[j1 + 2] + a[j3 + 2];
    x2i = a[j1 + 3] + a[j3 + 3];
    x3r = a[j1 + 2] - a[j3 + 2];
    x3i = a[j1 + 3] - a[j3 + 3];
    a[j0 + 2] = x0r + x2r;
    a[j0 + 3] = x0i - x2i;
    a[j1 + 2] = x0r - x2r;
    a[j1 + 3] = x0i + x2i;
    x0r = x1r + x3i;
    x0i = x1i + x3r;
    a[j2 + 2] = wk1i * x0r - wk1r * x0i;
    a[j2 + 3] = wk1i * x0i + wk1r * x0r;
    x0r = x1r - x3i;
    x0i = x1i - x3r;
    a[j3 + 2] = wk3i * x0r + wk3r * x0i;
    a[j3 + 3] = wk3i * x0i - wk3r * x0r;
}

fn cftrec4(n: usize, a: &mut [f64], nw: usize, w: &[f64]) {
    let mut m = n;
    while m > 512 {
        m >>= 2;
        cftmdl1(m, &mut a[(n - m)..], &w[(nw - (m >> 1))..]);
    }
    cftleaf(m, 1, &mut a[(n - m)..], nw, w);
    let mut k = 0;
    let mut isplt: i32;
    let mut j = n;
    while j > m {
        j -= m;
        k += 1;
        isplt = cfttree(m, j, k, a, nw, w);
        cftleaf(m, isplt, &mut a[(j - m)..], nw, w);
    }
}

fn cfttree(n: usize, j: usize, k: i32, a: &mut [f64], nw: usize, w: &[f64]) -> i32 {
    let isplt: i32;

    if (k & 3) != 0 {
        isplt = k & 1;
        if isplt != 0 {
            cftmdl1(n, &mut a[(j - n)..], &w[(nw - (n >> 1))..]);
        } else {
            cftmdl2(n, &mut a[(j - n)..], &w[(nw - n)..]);
        }
    } else {
        let mut m = n;
        let mut i = k;
        while (i & 3) == 0 {
            m <<= 2;
            i >>= 2;
        }
        isplt = i & 1;
        if isplt != 0 {
            while m > 128 {
                cftmdl1(m, &mut a[(j - m)..], &w[(nw - (m >> 1))..]);
                m >>= 2;
            }
        } else {
            while m > 128 {
                cftmdl2(m, &mut a[(j - m)..], &w[(nw - m)..]);
                m >>= 2;
            }
        }
    }
    return isplt;
}

fn cftleaf(n: usize, isplt: i32, a: &mut [f64], nw: usize, w: &[f64]) {
    if n == 512 {
        cftmdl1(128, a, &w[nw - 64..]);
        cftf161(a, &w[nw - 8..]);
        cftf162(&mut a[32..], &w[nw - 32..]);
        cftf161(&mut a[64..], &w[nw - 8..]);
        cftf161(&mut a[96..], &w[nw - 8..]);
        cftmdl2(128, &mut a[128..], &w[nw - 128..]);
        cftf161(&mut a[128..], &w[nw - 8..]);
        cftf162(&mut a[160..], &w[nw - 32..]);
        cftf161(&mut a[192..], &w[nw - 8..]);
        cftf162(&mut a[224..], &w[nw - 32..]);
        cftmdl1(128, &mut a[256..], &w[nw - 64..]);
        cftf161(&mut a[256..], &w[nw - 8..]);
        cftf162(&mut a[288..], &w[nw - 32..]);
        cftf161(&mut a[320..], &w[nw - 8..]);
        cftf161(&mut a[352..], &w[nw - 8..]);
        if isplt != 0 {
            cftmdl1(128, &mut a[384..], &w[nw - 64..]);
            cftf161(&mut a[480..], &w[nw - 8..]);
        } else {
            cftmdl2(128, &mut a[384..], &w[nw - 128..]);
            cftf162(&mut a[480..], &w[nw - 32..]);
        }
        cftf161(&mut a[384..], &w[nw - 8..]);
        cftf162(&mut a[416..], &w[nw - 32..]);
        cftf161(&mut a[448..], &w[nw - 8..]);
    } else {
        cftmdl1(64, a, &w[nw - 32..]);
        cftf081(a, &w[nw - 8..]);
        cftf082(&mut a[16..], &w[nw - 8..]);
        cftf081(&mut a[32..], &w[nw - 8..]);
        cftf081(&mut a[48..], &w[nw - 8..]);
        cftmdl2(64, &mut a[64..], &w[nw - 64..]);
        cftf081(&mut a[64..], &w[nw - 8..]);
        cftf082(&mut a[80..], &w[nw - 8..]);
        cftf081(&mut a[96..], &w[nw - 8..]);
        cftf082(&mut a[112..], &w[nw - 8..]);
        cftmdl1(64, &mut a[128..], &w[nw - 32..]);
        cftf081(&mut a[128..], &w[nw - 8..]);
        cftf082(&mut a[144..], &w[nw - 8..]);
        cftf081(&mut a[160..], &w[nw - 8..]);
        cftf081(&mut a[176..], &w[nw - 8..]);
        if isplt != 0 {
            cftmdl1(64, &mut a[192..], &w[nw - 32..]);
            cftf081(&mut a[240..], &w[nw - 8..]);
        } else {
            cftmdl2(64, &mut a[192..], &w[nw - 64..]);
            cftf082(&mut a[240..], &w[nw - 8..]);
        }
        cftf081(&mut a[192..], &w[nw - 8..]);
        cftf082(&mut a[208..], &w[nw - 8..]);
        cftf081(&mut a[224..], &w[nw - 8..]);
    }
}

fn cftmdl1(n: usize, a: &mut [f64], w: &[f64]) {
    let mut wk1r;
    let mut wk1i;
    let mut wk3r;
    let mut wk3i;
    let mut j0;

    let mh = n >> 3;
    let m = 2 * mh;
    let mut j1 = m;
    let mut j2 = j1 + m;
    let mut j3 = j2 + m;
    let mut x0r = a[0] + a[j2];
    let mut x0i = a[1] + a[j2 + 1];
    let mut x1r = a[0] - a[j2];
    let mut x1i = a[1] - a[j2 + 1];
    let mut x2r = a[j1] + a[j3];
    let mut x2i = a[j1 + 1] + a[j3 + 1];
    let mut x3r = a[j1] - a[j3];
    let mut x3i = a[j1 + 1] - a[j3 + 1];
    a[0] = x0r + x2r;
    a[1] = x0i + x2i;
    a[j1] = x0r - x2r;
    a[j1 + 1] = x0i - x2i;
    a[j2] = x1r - x3i;
    a[j2 + 1] = x1i + x3r;
    a[j3] = x1r + x3i;
    a[j3 + 1] = x1i - x3r;
    let wn4r = w[1];
    let mut k = 0;
    for j in (2..mh).step_by(2) {
        k += 4;
        wk1r = w[k];
        wk1i = w[k + 1];
        wk3r = w[k + 2];
        wk3i = w[k + 3];
        j1 = j + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[j] + a[j2];
        x0i = a[j + 1] + a[j2 + 1];
        x1r = a[j] - a[j2];
        x1i = a[j + 1] - a[j2 + 1];
        x2r = a[j1] + a[j3];
        x2i = a[j1 + 1] + a[j3 + 1];
        x3r = a[j1] - a[j3];
        x3i = a[j1 + 1] - a[j3 + 1];
        a[j] = x0r + x2r;
        a[j + 1] = x0i + x2i;
        a[j1] = x0r - x2r;
        a[j1 + 1] = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a[j2] = wk1r * x0r - wk1i * x0i;
        a[j2 + 1] = wk1r * x0i + wk1i * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a[j3] = wk3r * x0r + wk3i * x0i;
        a[j3 + 1] = wk3r * x0i - wk3i * x0r;
        j0 = m - j;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[j0] + a[j2];
        x0i = a[j0 + 1] + a[j2 + 1];
        x1r = a[j0] - a[j2];
        x1i = a[j0 + 1] - a[j2 + 1];
        x2r = a[j1] + a[j3];
        x2i = a[j1 + 1] + a[j3 + 1];
        x3r = a[j1] - a[j3];
        x3i = a[j1 + 1] - a[j3 + 1];
        a[j0] = x0r + x2r;
        a[j0 + 1] = x0i + x2i;
        a[j1] = x0r - x2r;
        a[j1 + 1] = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a[j2] = wk1i * x0r - wk1r * x0i;
        a[j2 + 1] = wk1i * x0i + wk1r * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a[j3] = wk3i * x0r + wk3r * x0i;
        a[j3 + 1] = wk3i * x0i - wk3r * x0r;
    }
    j0 = mh;
    j1 = j0 + m;
    j2 = j1 + m;
    j3 = j2 + m;
    x0r = a[j0] + a[j2];
    x0i = a[j0 + 1] + a[j2 + 1];
    x1r = a[j0] - a[j2];
    x1i = a[j0 + 1] - a[j2 + 1];
    x2r = a[j1] + a[j3];
    x2i = a[j1 + 1] + a[j3 + 1];
    x3r = a[j1] - a[j3];
    x3i = a[j1 + 1] - a[j3 + 1];
    a[j0] = x0r + x2r;
    a[j0 + 1] = x0i + x2i;
    a[j1] = x0r - x2r;
    a[j1 + 1] = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    a[j2] = wn4r * (x0r - x0i);
    a[j2 + 1] = wn4r * (x0i + x0r);
    x0r = x1r + x3i;
    x0i = x1i - x3r;
    a[j3] = -wn4r * (x0r + x0i);
    a[j3 + 1] = -wn4r * (x0i - x0r);
}

fn cftmdl2(n: usize, a: &mut [f64], w: &[f64]) {
    let mut wk1r;
    let mut wk1i;
    let mut wk3r;
    let mut wk3i;

    let mut wd1i;
    let mut wd1r;
    let mut wd3i;
    let mut wd3r;

    let mut y2r;
    let mut y2i;

    let mut j0;

    let mh = n >> 3;
    let m = 2 * mh;
    let wn4r = w[1];
    let mut j1 = m;
    let mut j2 = j1 + m;
    let mut j3 = j2 + m;
    let mut x0r = a[0] - a[j2 + 1];
    let mut x0i = a[1] + a[j2];
    let mut x1r = a[0] + a[j2 + 1];
    let mut x1i = a[1] - a[j2];
    let mut x2r = a[j1] - a[j3 + 1];
    let mut x2i = a[j1 + 1] + a[j3];
    let mut x3r = a[j1] + a[j3 + 1];
    let mut x3i = a[j1 + 1] - a[j3];
    let mut y0r = wn4r * (x2r - x2i);
    let mut y0i = wn4r * (x2i + x2r);
    a[0] = x0r + y0r;
    a[1] = x0i + y0i;
    a[j1] = x0r - y0r;
    a[j1 + 1] = x0i - y0i;
    y0r = wn4r * (x3r - x3i);
    y0i = wn4r * (x3i + x3r);
    a[j2] = x1r - y0i;
    a[j2 + 1] = x1i + y0r;
    a[j3] = x1r + y0i;
    a[j3 + 1] = x1i - y0r;
    let mut k = 0;
    let mut kr = 2 * m;
    for j in (2..mh).step_by(2) {
        k += 4;
        wk1r = w[k];
        wk1i = w[k + 1];
        wk3r = w[k + 2];
        wk3i = w[k + 3];
        kr -= 4;
        wd1i = w[kr];
        wd1r = w[kr + 1];
        wd3i = w[kr + 2];
        wd3r = w[kr + 3];
        j1 = j + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[j] - a[j2 + 1];
        x0i = a[j + 1] + a[j2];
        x1r = a[j] + a[j2 + 1];
        x1i = a[j + 1] - a[j2];
        x2r = a[j1] - a[j3 + 1];
        x2i = a[j1 + 1] + a[j3];
        x3r = a[j1] + a[j3 + 1];
        x3i = a[j1 + 1] - a[j3];
        y0r = wk1r * x0r - wk1i * x0i;
        y0i = wk1r * x0i + wk1i * x0r;
        y2r = wd1r * x2r - wd1i * x2i;
        y2i = wd1r * x2i + wd1i * x2r;
        a[j] = y0r + y2r;
        a[j + 1] = y0i + y2i;
        a[j1] = y0r - y2r;
        a[j1 + 1] = y0i - y2i;
        y0r = wk3r * x1r + wk3i * x1i;
        y0i = wk3r * x1i - wk3i * x1r;
        y2r = wd3r * x3r + wd3i * x3i;
        y2i = wd3r * x3i - wd3i * x3r;
        a[j2] = y0r + y2r;
        a[j2 + 1] = y0i + y2i;
        a[j3] = y0r - y2r;
        a[j3 + 1] = y0i - y2i;
        j0 = m - j;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[j0] - a[j2 + 1];
        x0i = a[j0 + 1] + a[j2];
        x1r = a[j0] + a[j2 + 1];
        x1i = a[j0 + 1] - a[j2];
        x2r = a[j1] - a[j3 + 1];
        x2i = a[j1 + 1] + a[j3];
        x3r = a[j1] + a[j3 + 1];
        x3i = a[j1 + 1] - a[j3];
        y0r = wd1i * x0r - wd1r * x0i;
        y0i = wd1i * x0i + wd1r * x0r;
        y2r = wk1i * x2r - wk1r * x2i;
        y2i = wk1i * x2i + wk1r * x2r;
        a[j0] = y0r + y2r;
        a[j0 + 1] = y0i + y2i;
        a[j1] = y0r - y2r;
        a[j1 + 1] = y0i - y2i;
        y0r = wd3i * x1r + wd3r * x1i;
        y0i = wd3i * x1i - wd3r * x1r;
        y2r = wk3i * x3r + wk3r * x3i;
        y2i = wk3i * x3i - wk3r * x3r;
        a[j2] = y0r + y2r;
        a[j2 + 1] = y0i + y2i;
        a[j3] = y0r - y2r;
        a[j3 + 1] = y0i - y2i;
    }
    wk1r = w[m];
    wk1i = w[m + 1];
    j0 = mh;
    j1 = j0 + m;
    j2 = j1 + m;
    j3 = j2 + m;
    x0r = a[j0] - a[j2 + 1];
    x0i = a[j0 + 1] + a[j2];
    x1r = a[j0] + a[j2 + 1];
    x1i = a[j0 + 1] - a[j2];
    x2r = a[j1] - a[j3 + 1];
    x2i = a[j1 + 1] + a[j3];
    x3r = a[j1] + a[j3 + 1];
    x3i = a[j1 + 1] - a[j3];
    y0r = wk1r * x0r - wk1i * x0i;
    y0i = wk1r * x0i + wk1i * x0r;
    y2r = wk1i * x2r - wk1r * x2i;
    y2i = wk1i * x2i + wk1r * x2r;
    a[j0] = y0r + y2r;
    a[j0 + 1] = y0i + y2i;
    a[j1] = y0r - y2r;
    a[j1 + 1] = y0i - y2i;
    y0r = wk1i * x1r - wk1r * x1i;
    y0i = wk1i * x1i + wk1r * x1r;
    y2r = wk1r * x3r - wk1i * x3i;
    y2i = wk1r * x3i + wk1i * x3r;
    a[j2] = y0r - y2r;
    a[j2 + 1] = y0i - y2i;
    a[j3] = y0r + y2r;
    a[j3 + 1] = y0i + y2i;
}

fn cftfx41(n: usize, a: &mut [f64], nw: usize, w: &[f64]) {
    if n == 128 {
        cftf161(a, &w[nw - 8..]);
        cftf162(&mut a[32..], &w[nw - 32..]);
        cftf161(&mut a[64..], &w[nw - 8..]);
        cftf161(&mut a[96..], &w[nw - 8..]);
    } else {
        cftf081(a, &w[nw - 8..]);
        cftf082(&mut a[16..], &w[nw - 8..]);
        cftf081(&mut a[32..], &w[nw - 8..]);
        cftf081(&mut a[48..], &w[nw - 8..]);
    }
}

fn cftf161(a: &mut [f64], w: &[f64]) {
    let wn4r = w[1];
    let wk1r = w[2];
    let wk1i = w[3];
    let mut x0r = a[0] + a[16];
    let mut x0i = a[1] + a[17];
    let mut x1r = a[0] - a[16];
    let mut x1i = a[1] - a[17];
    let mut x2r = a[8] + a[24];
    let mut x2i = a[9] + a[25];
    let mut x3r = a[8] - a[24];
    let mut x3i = a[9] - a[25];
    let y0r = x0r + x2r;
    let y0i = x0i + x2i;
    let y4r = x0r - x2r;
    let y4i = x0i - x2i;
    let y8r = x1r - x3i;
    let y8i = x1i + x3r;
    let y12r = x1r + x3i;
    let y12i = x1i - x3r;
    x0r = a[2] + a[18];
    x0i = a[3] + a[19];
    x1r = a[2] - a[18];
    x1i = a[3] - a[19];
    x2r = a[10] + a[26];
    x2i = a[11] + a[27];
    x3r = a[10] - a[26];
    x3i = a[11] - a[27];
    let y1r = x0r + x2r;
    let y1i = x0i + x2i;
    let y5r = x0r - x2r;
    let y5i = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    let y9r = wk1r * x0r - wk1i * x0i;
    let y9i = wk1r * x0i + wk1i * x0r;
    x0r = x1r + x3i;
    x0i = x1i - x3r;
    let y13r = wk1i * x0r - wk1r * x0i;
    let y13i = wk1i * x0i + wk1r * x0r;
    x0r = a[4] + a[20];
    x0i = a[5] + a[21];
    x1r = a[4] - a[20];
    x1i = a[5] - a[21];
    x2r = a[12] + a[28];
    x2i = a[13] + a[29];
    x3r = a[12] - a[28];
    x3i = a[13] - a[29];
    let y2r = x0r + x2r;
    let y2i = x0i + x2i;
    let y6r = x0r - x2r;
    let y6i = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    let y10r = wn4r * (x0r - x0i);
    let y10i = wn4r * (x0i + x0r);
    x0r = x1r + x3i;
    x0i = x1i - x3r;
    let y14r = wn4r * (x0r + x0i);
    let y14i = wn4r * (x0i - x0r);
    x0r = a[6] + a[22];
    x0i = a[7] + a[23];
    x1r = a[6] - a[22];
    x1i = a[7] - a[23];
    x2r = a[14] + a[30];
    x2i = a[15] + a[31];
    x3r = a[14] - a[30];
    x3i = a[15] - a[31];
    let y3r = x0r + x2r;
    let y3i = x0i + x2i;
    let y7r = x0r - x2r;
    let y7i = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    let y11r = wk1i * x0r - wk1r * x0i;
    let y11i = wk1i * x0i + wk1r * x0r;
    x0r = x1r + x3i;
    x0i = x1i - x3r;
    let y15r = wk1r * x0r - wk1i * x0i;
    let y15i = wk1r * x0i + wk1i * x0r;
    x0r = y12r - y14r;
    x0i = y12i - y14i;
    x1r = y12r + y14r;
    x1i = y12i + y14i;
    x2r = y13r - y15r;
    x2i = y13i - y15i;
    x3r = y13r + y15r;
    x3i = y13i + y15i;
    a[24] = x0r + x2r;
    a[25] = x0i + x2i;
    a[26] = x0r - x2r;
    a[27] = x0i - x2i;
    a[28] = x1r - x3i;
    a[29] = x1i + x3r;
    a[30] = x1r + x3i;
    a[31] = x1i - x3r;
    x0r = y8r + y10r;
    x0i = y8i + y10i;
    x1r = y8r - y10r;
    x1i = y8i - y10i;
    x2r = y9r + y11r;
    x2i = y9i + y11i;
    x3r = y9r - y11r;
    x3i = y9i - y11i;
    a[16] = x0r + x2r;
    a[17] = x0i + x2i;
    a[18] = x0r - x2r;
    a[19] = x0i - x2i;
    a[20] = x1r - x3i;
    a[21] = x1i + x3r;
    a[22] = x1r + x3i;
    a[23] = x1i - x3r;
    x0r = y5r - y7i;
    x0i = y5i + y7r;
    x2r = wn4r * (x0r - x0i);
    x2i = wn4r * (x0i + x0r);
    x0r = y5r + y7i;
    x0i = y5i - y7r;
    x3r = wn4r * (x0r - x0i);
    x3i = wn4r * (x0i + x0r);
    x0r = y4r - y6i;
    x0i = y4i + y6r;
    x1r = y4r + y6i;
    x1i = y4i - y6r;
    a[8] = x0r + x2r;
    a[9] = x0i + x2i;
    a[10] = x0r - x2r;
    a[11] = x0i - x2i;
    a[12] = x1r - x3i;
    a[13] = x1i + x3r;
    a[14] = x1r + x3i;
    a[15] = x1i - x3r;
    x0r = y0r + y2r;
    x0i = y0i + y2i;
    x1r = y0r - y2r;
    x1i = y0i - y2i;
    x2r = y1r + y3r;
    x2i = y1i + y3i;
    x3r = y1r - y3r;
    x3i = y1i - y3i;
    a[0] = x0r + x2r;
    a[1] = x0i + x2i;
    a[2] = x0r - x2r;
    a[3] = x0i - x2i;
    a[4] = x1r - x3i;
    a[5] = x1i + x3r;
    a[6] = x1r + x3i;
    a[7] = x1i - x3r;
}

fn cftf162(a: &mut [f64], w: &[f64]) {
    let wn4r = w[1];
    let wk1r = w[4];
    let wk1i = w[5];
    let wk3r = w[6];
    let wk3i = -w[7];
    let wk2r = w[8];
    let wk2i = w[9];
    let mut x1r = a[0] - a[17];
    let mut x1i = a[1] + a[16];
    let mut x0r = a[8] - a[25];
    let mut x0i = a[9] + a[24];
    let mut x2r = wn4r * (x0r - x0i);
    let mut x2i = wn4r * (x0i + x0r);
    let y0r = x1r + x2r;
    let y0i = x1i + x2i;
    let y4r = x1r - x2r;
    let y4i = x1i - x2i;
    x1r = a[0] + a[17];
    x1i = a[1] - a[16];
    x0r = a[8] + a[25];
    x0i = a[9] - a[24];
    x2r = wn4r * (x0r - x0i);
    x2i = wn4r * (x0i + x0r);
    let y8r = x1r - x2i;
    let y8i = x1i + x2r;
    let y12r = x1r + x2i;
    let y12i = x1i - x2r;
    x0r = a[2] - a[19];
    x0i = a[3] + a[18];
    x1r = wk1r * x0r - wk1i * x0i;
    x1i = wk1r * x0i + wk1i * x0r;
    x0r = a[10] - a[27];
    x0i = a[11] + a[26];
    x2r = wk3i * x0r - wk3r * x0i;
    x2i = wk3i * x0i + wk3r * x0r;
    let y1r = x1r + x2r;
    let y1i = x1i + x2i;
    let y5r = x1r - x2r;
    let y5i = x1i - x2i;
    x0r = a[2] + a[19];
    x0i = a[3] - a[18];
    x1r = wk3r * x0r - wk3i * x0i;
    x1i = wk3r * x0i + wk3i * x0r;
    x0r = a[10] + a[27];
    x0i = a[11] - a[26];
    x2r = wk1r * x0r + wk1i * x0i;
    x2i = wk1r * x0i - wk1i * x0r;
    let y9r = x1r - x2r;
    let y9i = x1i - x2i;
    let y13r = x1r + x2r;
    let y13i = x1i + x2i;
    x0r = a[4] - a[21];
    x0i = a[5] + a[20];
    x1r = wk2r * x0r - wk2i * x0i;
    x1i = wk2r * x0i + wk2i * x0r;
    x0r = a[12] - a[29];
    x0i = a[13] + a[28];
    x2r = wk2i * x0r - wk2r * x0i;
    x2i = wk2i * x0i + wk2r * x0r;
    let y2r = x1r + x2r;
    let y2i = x1i + x2i;
    let y6r = x1r - x2r;
    let y6i = x1i - x2i;
    x0r = a[4] + a[21];
    x0i = a[5] - a[20];
    x1r = wk2i * x0r - wk2r * x0i;
    x1i = wk2i * x0i + wk2r * x0r;
    x0r = a[12] + a[29];
    x0i = a[13] - a[28];
    x2r = wk2r * x0r - wk2i * x0i;
    x2i = wk2r * x0i + wk2i * x0r;
    let y10r = x1r - x2r;
    let y10i = x1i - x2i;
    let y14r = x1r + x2r;
    let y14i = x1i + x2i;
    x0r = a[6] - a[23];
    x0i = a[7] + a[22];
    x1r = wk3r * x0r - wk3i * x0i;
    x1i = wk3r * x0i + wk3i * x0r;
    x0r = a[14] - a[31];
    x0i = a[15] + a[30];
    x2r = wk1i * x0r - wk1r * x0i;
    x2i = wk1i * x0i + wk1r * x0r;
    let y3r = x1r + x2r;
    let y3i = x1i + x2i;
    let y7r = x1r - x2r;
    let y7i = x1i - x2i;
    x0r = a[6] + a[23];
    x0i = a[7] - a[22];
    x1r = wk1i * x0r + wk1r * x0i;
    x1i = wk1i * x0i - wk1r * x0r;
    x0r = a[14] + a[31];
    x0i = a[15] - a[30];
    x2r = wk3i * x0r - wk3r * x0i;
    x2i = wk3i * x0i + wk3r * x0r;
    let y11r = x1r + x2r;
    let y11i = x1i + x2i;
    let y15r = x1r - x2r;
    let y15i = x1i - x2i;
    x1r = y0r + y2r;
    x1i = y0i + y2i;
    x2r = y1r + y3r;
    x2i = y1i + y3i;
    a[0] = x1r + x2r;
    a[1] = x1i + x2i;
    a[2] = x1r - x2r;
    a[3] = x1i - x2i;
    x1r = y0r - y2r;
    x1i = y0i - y2i;
    x2r = y1r - y3r;
    x2i = y1i - y3i;
    a[4] = x1r - x2i;
    a[5] = x1i + x2r;
    a[6] = x1r + x2i;
    a[7] = x1i - x2r;
    x1r = y4r - y6i;
    x1i = y4i + y6r;
    x0r = y5r - y7i;
    x0i = y5i + y7r;
    x2r = wn4r * (x0r - x0i);
    x2i = wn4r * (x0i + x0r);
    a[8] = x1r + x2r;
    a[9] = x1i + x2i;
    a[10] = x1r - x2r;
    a[11] = x1i - x2i;
    x1r = y4r + y6i;
    x1i = y4i - y6r;
    x0r = y5r + y7i;
    x0i = y5i - y7r;
    x2r = wn4r * (x0r - x0i);
    x2i = wn4r * (x0i + x0r);
    a[12] = x1r - x2i;
    a[13] = x1i + x2r;
    a[14] = x1r + x2i;
    a[15] = x1i - x2r;
    x1r = y8r + y10r;
    x1i = y8i + y10i;
    x2r = y9r - y11r;
    x2i = y9i - y11i;
    a[16] = x1r + x2r;
    a[17] = x1i + x2i;
    a[18] = x1r - x2r;
    a[19] = x1i - x2i;
    x1r = y8r - y10r;
    x1i = y8i - y10i;
    x2r = y9r + y11r;
    x2i = y9i + y11i;
    a[20] = x1r - x2i;
    a[21] = x1i + x2r;
    a[22] = x1r + x2i;
    a[23] = x1i - x2r;
    x1r = y12r - y14i;
    x1i = y12i + y14r;
    x0r = y13r + y15i;
    x0i = y13i - y15r;
    x2r = wn4r * (x0r - x0i);
    x2i = wn4r * (x0i + x0r);
    a[24] = x1r + x2r;
    a[25] = x1i + x2i;
    a[26] = x1r - x2r;
    a[27] = x1i - x2i;
    x1r = y12r + y14i;
    x1i = y12i - y14r;
    x0r = y13r - y15i;
    x0i = y13i + y15r;
    x2r = wn4r * (x0r - x0i);
    x2i = wn4r * (x0i + x0r);
    a[28] = x1r - x2i;
    a[29] = x1i + x2r;
    a[30] = x1r + x2i;
    a[31] = x1i - x2r;
}

fn cftf081(a: &mut [f64], w: &[f64]) {
    let wn4r = w[1];
    let mut x0r = a[0] + a[8];
    let mut x0i = a[1] + a[9];
    let mut x1r = a[0] - a[8];
    let mut x1i = a[1] - a[9];
    let mut x2r = a[4] + a[12];
    let mut x2i = a[5] + a[13];
    let mut x3r = a[4] - a[12];
    let mut x3i = a[5] - a[13];
    let y0r = x0r + x2r;
    let y0i = x0i + x2i;
    let y2r = x0r - x2r;
    let y2i = x0i - x2i;
    let y1r = x1r - x3i;
    let y1i = x1i + x3r;
    let y3r = x1r + x3i;
    let y3i = x1i - x3r;
    x0r = a[2] + a[10];
    x0i = a[3] + a[11];
    x1r = a[2] - a[10];
    x1i = a[3] - a[11];
    x2r = a[6] + a[14];
    x2i = a[7] + a[15];
    x3r = a[6] - a[14];
    x3i = a[7] - a[15];
    let y4r = x0r + x2r;
    let y4i = x0i + x2i;
    let y6r = x0r - x2r;
    let y6i = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    x2r = x1r + x3i;
    x2i = x1i - x3r;
    let y5r = wn4r * (x0r - x0i);
    let y5i = wn4r * (x0r + x0i);
    let y7r = wn4r * (x2r - x2i);
    let y7i = wn4r * (x2r + x2i);
    a[8] = y1r + y5r;
    a[9] = y1i + y5i;
    a[10] = y1r - y5r;
    a[11] = y1i - y5i;
    a[12] = y3r - y7i;
    a[13] = y3i + y7r;
    a[14] = y3r + y7i;
    a[15] = y3i - y7r;
    a[0] = y0r + y4r;
    a[1] = y0i + y4i;
    a[2] = y0r - y4r;
    a[3] = y0i - y4i;
    a[4] = y2r - y6i;
    a[5] = y2i + y6r;
    a[6] = y2r + y6i;
    a[7] = y2i - y6r;
}

fn cftf082(a: &mut [f64], w: &[f64]) {
    let wn4r = w[1];
    let wk1r = w[2];
    let wk1i = w[3];
    let y0r = a[0] - a[9];
    let y0i = a[1] + a[8];
    let y1r = a[0] + a[9];
    let y1i = a[1] - a[8];
    let mut x0r = a[4] - a[13];
    let mut x0i = a[5] + a[12];
    let y2r = wn4r * (x0r - x0i);
    let y2i = wn4r * (x0i + x0r);
    x0r = a[4] + a[13];
    x0i = a[5] - a[12];
    let y3r = wn4r * (x0r - x0i);
    let y3i = wn4r * (x0i + x0r);
    x0r = a[2] - a[11];
    x0i = a[3] + a[10];
    let y4r = wk1r * x0r - wk1i * x0i;
    let y4i = wk1r * x0i + wk1i * x0r;
    x0r = a[2] + a[11];
    x0i = a[3] - a[10];
    let y5r = wk1i * x0r - wk1r * x0i;
    let y5i = wk1i * x0i + wk1r * x0r;
    x0r = a[6] - a[15];
    x0i = a[7] + a[14];
    let y6r = wk1i * x0r - wk1r * x0i;
    let y6i = wk1i * x0i + wk1r * x0r;
    x0r = a[6] + a[15];
    x0i = a[7] - a[14];
    let y7r = wk1r * x0r - wk1i * x0i;
    let y7i = wk1r * x0i + wk1i * x0r;
    x0r = y0r + y2r;
    x0i = y0i + y2i;
    let mut x1r = y4r + y6r;
    let mut x1i = y4i + y6i;
    a[0] = x0r + x1r;
    a[1] = x0i + x1i;
    a[2] = x0r - x1r;
    a[3] = x0i - x1i;
    x0r = y0r - y2r;
    x0i = y0i - y2i;
    x1r = y4r - y6r;
    x1i = y4i - y6i;
    a[4] = x0r - x1i;
    a[5] = x0i + x1r;
    a[6] = x0r + x1i;
    a[7] = x0i - x1r;
    x0r = y1r - y3i;
    x0i = y1i + y3r;
    x1r = y5r - y7r;
    x1i = y5i - y7i;
    a[8] = x0r + x1r;
    a[9] = x0i + x1i;
    a[10] = x0r - x1r;
    a[11] = x0i - x1i;
    x0r = y1r + y3i;
    x0i = y1i - y3r;
    x1r = y5r + y7r;
    x1i = y5i + y7i;
    a[12] = x0r - x1i;
    a[13] = x0i + x1r;
    a[14] = x0r + x1i;
    a[15] = x0i - x1r;
}

fn cftf040(a: &mut [f64]) {
    let x0r = a[0] + a[4];
    let x0i = a[1] + a[5];
    let x1r = a[0] - a[4];
    let x1i = a[1] - a[5];
    let x2r = a[2] + a[6];
    let x2i = a[3] + a[7];
    let x3r = a[2] - a[6];
    let x3i = a[3] - a[7];
    a[0] = x0r + x2r;
    a[1] = x0i + x2i;
    a[2] = x1r - x3i;
    a[3] = x1i + x3r;
    a[4] = x0r - x2r;
    a[5] = x0i - x2i;
    a[6] = x1r + x3i;
    a[7] = x1i - x3r;
}

fn cftb040(a: &mut [f64]) {
    let x0r = a[0] + a[4];
    let x0i = a[1] + a[5];
    let x1r = a[0] - a[4];
    let x1i = a[1] - a[5];
    let x2r = a[2] + a[6];
    let x2i = a[3] + a[7];
    let x3r = a[2] - a[6];
    let x3i = a[3] - a[7];
    a[0] = x0r + x2r;
    a[1] = x0i + x2i;
    a[2] = x1r + x3i;
    a[3] = x1i - x3r;
    a[4] = x0r - x2r;
    a[5] = x0i - x2i;
    a[6] = x1r - x3i;
    a[7] = x1i + x3r;
}

fn cftx020(a: &mut [f64]) {
    let x0r = a[0] - a[2];
    let x0i = a[1] - a[3];
    a[0] += a[2];
    a[1] += a[3];
    a[2] = x0r;
    a[3] = x0i;
}

fn rftfsub(n: usize, a: &mut [f64], nc: usize, c: &[f64]) {
    let m = n >> 1;
    let ks = 2 * nc / m;
    let mut kk = 0;
    for j in (2..m).step_by(2) {
        let k = n - j;
        kk += ks;
        let wkr = 0.5 - c[nc - kk];
        let wki = c[kk];
        let xr = a[j] - a[k];
        let xi = a[j + 1] + a[k + 1];
        let yr = wkr * xr - wki * xi;
        let yi = wkr * xi + wki * xr;
        a[j] -= yr;
        a[j + 1] -= yi;
        a[k] += yr;
        a[k + 1] -= yi;
    }
}

fn rftbsub(n: usize, a: &mut [f64], nc: usize, c: &[f64]) {
    let m = n >> 1;
    let ks = 2 * nc / m;
    let mut kk = 0;
    for j in (2..m).step_by(2) {
        let k = n - j;
        kk += ks;
        let wkr = 0.5 - c[nc - kk];
        let wki = c[kk];
        let xr = a[j] - a[k];
        let xi = a[j + 1] + a[k + 1];
        let yr = wkr * xr + wki * xi;
        let yi = wkr * xi - wki * xr;
        a[j] -= yr;
        a[j + 1] -= yi;
        a[k] += yr;
        a[k + 1] -= yi;
    }
}

#[allow(dead_code)]
fn dctsub(n: usize, a: &mut [f64], nc: usize, c: &[f64]) {
    let m = n >> 1;
    let ks = nc / n;
    let mut kk = 0;
    for j in 1..m {
        let k = n - j;
        kk += ks;
        let wkr = c[kk] - c[nc - kk];
        let wki = c[kk] + c[nc - kk];
        let xr = wki * a[j] - wkr * a[k];
        a[j] = wkr * a[j] + wki * a[k];
        a[k] = xr;
    }
    a[m] *= c[0];
}

#[allow(dead_code)]
fn dstsub(n: usize, a: &mut [f64], nc: usize, c: &[f64]) {
    let m = n >> 1;
    let ks = nc / n;
    let mut kk = 0;
    for j in 1..m {
        let k = n - j;
        kk += ks;
        let wkr = c[kk] - c[nc - kk];
        let wki = c[kk] + c[nc - kk];
        let xr = wki * a[k] - wkr * a[j];
        a[k] = wkr * a[k] + wki * a[j];
        a[j] = xr;
    }
    a[m] *= c[0];
}
