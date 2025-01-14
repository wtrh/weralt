//! Matlab functions implemented for WORLD.
//! Since these functions are implemented as the same function of Matlab,
//! the source code does not follow the style guide (Names of variables
//! and functions).
//! Please see the reference of Matlab to show the usage of functions.

use crate::common::{FFTProsess, ForwardRealFFT, InverseRealFFT};
use crate::fft::fft_complex;

//---

/// `FilterForDecimate()` calculates the coefficients of low-pass filter and
/// carries out the filtering. This function is only used for `decimate()`.
fn FilterForDecimate(x: &[f64], x_length: usize, r: usize, y: &mut [f64]) {
    // filter Coefficients
    let mut a: [f64; 3] = [0.0, 0.0, 0.0];
    let mut b: [f64; 2] = [0.0, 0.0];

    match r {
        11 => {
            // fs : 44100 (default)
            a[0] = 2.450743295230728;
            a[1] = -2.06794904601978;
            a[2] = 0.59574774438332101;
            b[0] = 0.0026822508007163792;
            b[1] = 0.0080467524021491377;
        }
        12 => {
            // fs : 48000
            a[0] = 2.4981398605924205;
            a[1] = -2.1368928194784025;
            a[2] = 0.62187513816221485;
            b[0] = 0.0021097275904709001;
            b[1] = 0.0063291827714127002;
        }
        10 => {
            a[0] = 2.3936475118069387;
            a[1] = -1.9873904075111861;
            a[2] = 0.5658879979027055;
            b[0] = 0.0034818622251927556;
            b[1] = 0.010445586675578267;
        }
        9 => {
            a[0] = 2.3236003491759578;
            a[1] = -1.8921545617463598;
            a[2] = 0.53148928133729068;
            b[0] = 0.0046331164041389372;
            b[1] = 0.013899349212416812;
        }
        8 => {
            // fs : 32000
            a[0] = 2.2357462340187593;
            a[1] = -1.7780899984041358;
            a[2] = 0.49152555365968692;
            b[0] = 0.0063522763407111993;
            b[1] = 0.019056829022133598;
        }
        7 => {
            a[0] = 2.1225239019534703;
            a[1] = -1.6395144861046302;
            a[2] = 0.44469707800587366;
            b[0] = 0.0090366882681608418;
            b[1] = 0.027110064804482525;
        }
        6 => {
            // fs : 24000 and 22050
            a[0] = 1.9715352749512141;
            a[1] = -1.4686795689225347;
            a[2] = 0.3893908434965701;
            b[0] = 0.013469181309343825;
            b[1] = 0.040407543928031475;
        }
        5 => {
            a[0] = 1.7610939654280557;
            a[1] = -1.2554914843859768;
            a[2] = 0.3237186507788215;
            b[0] = 0.021334858522387423;
            b[1] = 0.06400457556716227;
        }
        4 => {
            // fs : 16000
            a[0] = 1.4499664446880227;
            a[1] = -0.98943497080950582;
            a[2] = 0.24578252340690215;
            b[0] = 0.036710750339322612;
            b[1] = 0.11013225101796784;
        }
        3 => {
            a[0] = 0.95039378983237421;
            a[1] = -0.67429146741526791;
            a[2] = 0.15412211621346475;
            b[0] = 0.071221945171178636;
            b[1] = 0.21366583551353591;
        }
        2 => {
            // fs : 8000
            a[0] = 0.041156734567757189;
            a[1] = -0.42599112459189636;
            a[2] = 0.041037215479961225;
            b[0] = 0.16797464681802227;
            b[1] = 0.50392394045406674;
        }
        _ => {
            a[0] = 0.0;
            a[1] = 0.0;
            a[2] = 0.0;
            b[0] = 0.0;
            b[1] = 0.0;
        }
    }

    // Filtering on time domain.
    let mut w: [f64; 3] = [0.0, 0.0, 0.0];
    let mut wt: f64;
    for i in 0..x_length {
        wt = x[i] + a[0] * w[0] + a[1] * w[1] + a[2] * w[2];
        y[i] = b[0] * wt + b[1] * w[0] + b[1] * w[1] + b[0] * w[2];
        w[2] = w[1];
        w[1] = w[0];
        w[0] = wt;
    }
}

//---

/// `fftshift()` swaps the left and right halves of input vector.
/// http://www.mathworks.com/help/matlab/ref/fftshift.html
///
/// - Input
///     - `x`           : Input vector
///     - `x_length`    : Length of `x`
///
/// - Output
///     - `y`           : Swapped vector `x`
///
/// Caution:
///   Lengths of `index` and `edges` must be the same.
pub fn fftshift(x: &[f64], x_length: usize, y: &mut [f64]) {
    for i in 0..(x_length / 2) {
        y[i] = x[i + x_length / 2];
        y[i + x_length / 2] = x[i];
    }
}

/// `histc()` counts the number of values in vector x that fall between the
/// elements in the edges vector (which must contain monotonically
/// nondecreasing values). n is a length(edges) vector containing these counts.
/// No elements of x can be complex.
/// http://www.mathworks.co.jp/help/techdoc/ref/histc.html
///
/// - Input
///     - `x`               : Input vector
///     - `x_length`        : Length of `x`
///     - `edges`           : Input matrix (1-dimension)
///     - `edges_length`    : Length of `edges`
///
/// - Output
///     - `index`           : Result counted in vector `x`
/// Caution:
///   Lengths of `index` and `edges` must be the same.
pub fn histc(x: &[f64], x_length: usize, edges: &[f64], edges_length: usize, index: &mut [usize]) {
    let mut count = 1;

    let mut i = 0;
    while i < edges_length {
        index[i] = 1;
        if edges[i] >= x[0] {
            break;
        }
        i += 1;
    }
    while i < edges_length {
        if edges[i] < x[count] {
            index[i] = count;
        } else {
            index[i] = count;
            i -= 1;
            count += 1;
        }
        if count == x_length {
            break;
        }
        i += 1;
    }
    count -= 1;
    while i < edges_length {
        index[i] = count;
        i += 1;
    }
}

/// `interp1()` interpolates to find yi, the values of the underlying function
/// Y at the points in the vector or array xi. x must be a vector.
/// http://www.mathworks.co.jp/help/techdoc/ref/interp1.html
///
/// - Input
///     - `x`           : Input vector (Time axis)
///     - `y`           : Values at `x[n]`
///     - 'x_length`    : Length of `x` (Length of `y` must be the same)
///     - `xi`          : Required vector
///     - `xi_length`   : Length of `xi` (Length of `yi` must be the same)
///
/// - Output
///     - `yi`          : Interpolated vector
pub fn interp1(
    x: &[f64],
    y: &[f64],
    x_length: usize,
    xi: &[f64],
    xi_length: usize,
    yi: &mut [f64],
) {
    let mut h: Vec<f64> = vec![0.0; x_length - 1];
    let mut k: Vec<usize> = vec![0; xi_length];

    for i in 0..x_length - 1 {
        h[i] = x[i + 1] - x[i];
    }
    for i in 0..xi_length {
        k[i] = 0;
    }

    histc(x, x_length, xi, xi_length, &mut k);

    for i in 0..xi_length {
        let s = (xi[i] - x[k[i] - 1]) / h[k[i] - 1];
        yi[i] = y[k[i] - 1] + s * (y[k[i]] - y[k[i] - 1]);
    }
}

/// `decimate()` carries out down sampling by both IIR and FIR filters.
/// Filter coeffiencts are based on `FilterForDecimate()`.
///
/// - Input
///     - `x`           : Input signal
///     - `x_length`    : Length of `x`
///     - `r`           : Coefficient used for down sampling (fs after down sampling is fs/`r`)
///
/// - Output
///     - `y`           : Output signal
pub fn decimate(x: &[f64], x_length: usize, r: usize, y: &mut [f64]) {
    #[allow(non_upper_case_globals)]
    const kNFact: usize = 9;
    let mut tmp1: Vec<f64> = vec![0.0; x_length + kNFact * 2];
    let mut tmp2: Vec<f64> = vec![0.0; x_length + kNFact * 2];

    for i in 0..kNFact {
        tmp1[i] = 2.0 * x[0] - x[kNFact - i];
    }
    for i in kNFact..kNFact + x_length {
        tmp1[i] = x[i - kNFact];
    }
    for i in kNFact + x_length..2 * kNFact + x_length {
        tmp1[i] = 2.0 * x[x_length - 1] - x[x_length - 2 - (i - (kNFact + x_length))];
    }

    FilterForDecimate(&tmp1, 2 * kNFact + x_length, r, &mut tmp2);
    for i in 0..2 * kNFact + x_length {
        tmp1[i] = tmp2[2 * kNFact + x_length - i - 1];
    }
    FilterForDecimate(&tmp1, 2 * kNFact + x_length, r, &mut tmp2);
    for i in 0..2 * kNFact + x_length {
        tmp1[i] = tmp2[2 * kNFact + x_length - i - 1];
    }

    let nout = (x_length - 1) / r + 1;
    let nbeg = r + x_length - r * nout;

    let mut count = 0;
    let mut i = nbeg;
    while i < x_length + kNFact {
        y[count] = tmp1[i + kNFact - 1];
        count += 1;
        i += r;
    }
}

/// `matlab_round()` calculates rounding.
///
/// - Input
///     - `x`   : Input value
///
/// - Output
///     - `y`   : Rounded value
pub fn matlab_round(x: f64) -> i32 {
    return if x > 0.0 {
        (x + 0.5) as i32
    } else {
        (x - 0.5) as i32
    };
}

/// `diff()` calculates differences and approximate derivatives
/// http://www.mathworks.co.jp/help/techdoc/ref/diff.html
///
/// - Input
///     - `x`           : Input signal
///     - `x_length`    : Length of `x`
///
/// - Output
///     - `y`           : Output signal
pub fn diff(x: &[f64], x_length: usize, y: &mut [f64]) {
    for i in 0..x_length - 1 {
        y[i] = x[i + 1] - x[i];
    }
}

/// `interp1Q()` is the special case of `interp1()`.
/// We can use this function, provided that All periods of x-axis is the same.
///
/// - Input
///     - `x`           : Origin of the x-axis
///     - `shift`       : Period of the x-axis
///     - `y`           : Values at `x[n]`
///     - `x_length`    : Length of `x` (Length of `y` must be the same)
///     - `xi`          : Required vector
///     - `xi_length`   : Length of `xi` (Length of `yi` must be the same)
///
/// - Output
///     - `yi`          : Interpolated vector
///
/// Caution:
///   Length of `xi` and `yi` must be the same.
pub fn interp1Q(
    x: f64,
    shift: f64,
    y: &[f64],
    x_length: usize,
    xi: &[f64],
    xi_length: usize,
    yi: &mut [f64],
) {
    let mut xi_fraction: Vec<f64> = vec![0.0; xi_length];
    let mut delta_y: Vec<f64> = vec![0.0; x_length];
    let mut xi_base: Vec<usize> = vec![0; xi_length];

    let delta_x = shift;
    for i in 0..xi_length {
        xi_base[i] = ((xi[i] - x) / delta_x) as usize;
        xi_fraction[i] = (xi[i] - x) / delta_x - xi_base[i] as f64;
    }
    diff(y, x_length, &mut delta_y);
    delta_y[x_length - 1] = 0.0;

    for i in 0..xi_length {
        yi[i] = y[xi_base[i]] + delta_y[xi_base[i]] * xi_fraction[i];
    }
}

/// For generating (pseudo-)random number.
///
/// # Examples
///
/// ```
/// use weralt::Rng;
/// struct R {
///     x: u32,
/// }
///
/// impl Rng for R {
///     fn new() -> Self {
///         Self { x: 1 }
///     }
///     fn gen(&mut self) -> f64 {
///         const A: u32 = 48271;
///         const M: u32 = 0x7fffffff;
///         let r = A * self.x % M;
///         self.x = r;
///         return r as f64;
///     }
///     fn reseed(&mut self) {
///         self.x = 1;
///     }
/// }
///
/// let mut r = R::new();
/// let a = r.gen();
/// let b = r.gen();
/// r.reseed();
/// let c = r.gen();
/// assert_ne!(a, b);
/// assert_eq!(a, c);
/// ```
pub trait Rng {
    /// `new()` generates initialized structs for generating random number.
    fn new() -> Self
    where
        Self: Sized;
    /// `gen()` generates (pseudo-)random numbers.
    fn gen(&mut self) -> f64;
    /// `reseed()` forces to seed the RNG using initial values.
    fn reseed(&mut self);
}

const INIT_RANDN_X: u32 = 123456789;
const INIT_RANDN_Y: u32 = 362436069;
const INIT_RANDN_Z: u32 = 521288629;
const INIT_RANDN_W: u32 = 88675123;

/// Pseudo-RNG based on xorshift method.
pub struct Randn {
    x: u32,
    y: u32,
    z: u32,
    w: u32,
}

impl Rng for Randn {
    fn new() -> Self {
        return Self {
            x: INIT_RANDN_X,
            y: INIT_RANDN_Y,
            z: INIT_RANDN_Z,
            w: INIT_RANDN_W,
        };
    }
    fn gen(&mut self) -> f64 {
        let mut t = self.x ^ (self.x << 11);
        self.x = self.y;
        self.y = self.z;
        self.z = self.w;
        self.w = (self.w ^ (self.w >> 19)) ^ (t ^ (t >> 8));

        let mut tmp = self.w >> 4;
        for _ in 0..11 {
            t = self.x ^ (self.x << 11);
            self.x = self.y;
            self.y = self.z;
            self.z = self.w;
            self.w = (self.w ^ (self.w >> 19)) ^ (t ^ (t >> 8));
            tmp += self.w >> 4;
        }
        return tmp as f64 / 268435456.0 - 6.0;
    }
    fn reseed(&mut self) {
        self.x = INIT_RANDN_X;
        self.y = INIT_RANDN_Y;
        self.z = INIT_RANDN_Z;
        self.w = INIT_RANDN_W;
    }
}

/// `fast_fftfilt()` carries out the convolution on the frequency domain.
///
/// - Input:
///     - `x`                   : Input signal
///     - `x_length`            : Length of `x`
///     - `h`                   : Impulse response
///     - `h_length`            : Length of `h`
///     - `fft_size`            : Length of `FFT`
///     - `forward_real_fft`    : Struct to speed up the forward FFT
///     - `inverse_real_fft`    : Struct to speed up the inverse FFT
///
/// - Output
///     - `y`                   : Calculated result.
#[allow(dead_code)]
pub fn fast_fftfilt(
    x: &[f64],
    x_length: usize,
    h: &[f64],
    h_length: usize,
    fft_size: usize,
    forward_real_fft: &mut ForwardRealFFT,
    inverse_real_fft: &mut InverseRealFFT,
    y: &mut [f64],
) {
    let mut x_spectrum: Vec<fft_complex> = vec![fft_complex::default(); fft_size];

    for i in 0..x_length {
        forward_real_fft.waveform[i] = x[i] / fft_size as f64;
    }
    for i in x_length..fft_size {
        forward_real_fft.waveform[i] = 0.0;
    }
    forward_real_fft.exec();
    for i in 0..=fft_size / 2 {
        x_spectrum[i][0] = forward_real_fft.spectrum[i][0];
        x_spectrum[i][1] = forward_real_fft.spectrum[i][1];
    }

    for i in 0..h_length {
        forward_real_fft.waveform[i] = h[i] / fft_size as f64;
    }
    for i in h_length..fft_size {
        forward_real_fft.waveform[i] = 0.0;
    }
    forward_real_fft.exec();

    for i in 0..=fft_size / 2 {
        inverse_real_fft.spectrum[i][0] = x_spectrum[i][0] * forward_real_fft.spectrum[i][0]
            - x_spectrum[i][1] * forward_real_fft.spectrum[i][1];
        inverse_real_fft.spectrum[i][1] = x_spectrum[i][0] * forward_real_fft.spectrum[i][1]
            + x_spectrum[i][1] * forward_real_fft.spectrum[i][0];
    }
    inverse_real_fft.exec();

    for i in 0..fft_size {
        y[i] = inverse_real_fft.waveform[i];
    }
}

/// `matlab_std()` calculates the standard deviation of the input vector.
///
/// - Input
///     - `x`           : Input vector
///     - `x_length`    : Length of `x`
///
/// - Output
///   - Calculated standard deviation
#[allow(dead_code)]
pub fn matlab_std(x: &[f64], x_length: usize) -> f64 {
    let mut average = 0.0;
    for i in 0..x_length {
        average += x[i];
    }
    average /= x_length as f64;

    let mut s = 0.0;
    for i in 0..x_length {
        s += (x[i] - average).powf(2.0);
    }
    s /= x_length as f64 - 1.0;

    return s.sqrt();
}

#[cfg(test)]
mod tests {
    use super::*;

    struct R {
        x: u32,
    }

    impl Rng for R {
        fn new() -> Self {
            Self { x: 1 }
        }
        fn gen(&mut self) -> f64 {
            const A: u32 = 48271;
            const M: u32 = 0x7fffffff;
            let r = A * self.x % M;
            self.x = r;
            return r as f64;
        }
        fn reseed(&mut self) {
            self.x = 1;
        }
    }

    #[test]
    fn rand_num_trait() {
        let mut r = R::new();
        let a = r.gen();
        let b = r.gen();
        r.reseed();
        let c = r.gen();
        assert_ne!(a, b);
        assert_eq!(a, c);
    }
}
