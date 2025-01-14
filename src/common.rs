//! `common` module includes functions used in at least two files.
//! 1. Common functions
//! 2. FFT, IFFT and minimum phase analysis.
//!
//! Functions `MinimumPhaseAnalysis::get_minimum_phase_spectrum()`
//! calculate minimum phase spectrum.
//!
//! Forward and inverse FFT do not have the function `get*()`,
//! because forward FFT and inverse FFT can run in one step.

use crate::fft::{
    fft_complex, fft_execute, fft_plan, fft_plan_dft_1d, fft_plan_dft_c2r_1d, fft_plan_dft_r2c_1d,
    FFT_BACKWARD, FFT_ESTIMATE, FFT_FORWARD,
};

// Structs on FFT
/// Forward FFT in the real sequence
#[allow(dead_code)]
pub struct ForwardRealFFT<'a> {
    pub fft_size: usize,
    pub waveform: Vec<f64>,
    pub spectrum: Vec<fft_complex>,
    forward_fft: fft_plan<'a>,
}

/// Inverse FFT in the real sequence
#[allow(dead_code)]
pub struct InverseRealFFT<'a> {
    pub fft_size: usize,
    pub waveform: Vec<f64>,
    pub spectrum: Vec<fft_complex>,
    inverse_fft: fft_plan<'a>,
}

/// Inverse FFT in the complex sequence
#[allow(dead_code)]
pub struct InverseComplexFFT<'a> {
    pub fft_size: usize,
    pub input: Vec<fft_complex>,
    pub output: Vec<fft_complex>,
    inverse_fft: fft_plan<'a>,
}

/// Minimum phase analysis from logarithmic power spectrum
#[allow(dead_code)]
pub struct MinimumPhaseAnalysis<'a> {
    pub fft_size: usize,
    pub log_spectrum: Vec<f64>,
    pub minimum_phase_spectrum: Vec<fft_complex>,
    pub cepstrum: Vec<fft_complex>,
    inverse_fft: fft_plan<'a>,
    forward_fft: fft_plan<'a>,
}

use crate::constantnumbers::*;
use crate::matlabfunctions::*;

fn SetParametersForLinearSmoothing(
    boundary: usize,
    fft_size: usize,
    fs: i32,
    width: f64,
    power_spectrum: &[f64],
    mirroring_spectrum: &mut [f64],
    mirroring_segment: &mut [f64],
    frequency_axis: &mut [f64],
) {
    for i in 0..boundary {
        mirroring_spectrum[i] = power_spectrum[boundary - i];
    }
    for i in boundary..fft_size / 2 + boundary {
        mirroring_spectrum[i] = power_spectrum[i - boundary];
    }
    for i in fft_size / 2 + boundary..=fft_size / 2 + boundary * 2 {
        mirroring_spectrum[i] = power_spectrum[fft_size / 2 - (i - (fft_size / 2 + boundary))];
    }

    mirroring_segment[0] = mirroring_spectrum[0] * fs as f64 / fft_size as f64;
    for i in 1..fft_size / 2 + boundary * 2 + 1 {
        mirroring_segment[i] =
            mirroring_spectrum[i] * fs as f64 / fft_size as f64 + mirroring_segment[i - 1];
    }

    for i in 0..=fft_size / 2 {
        frequency_axis[i] = i as f64 / fft_size as f64 * fs as f64 - width / 2.0;
    }
}

//---

/// `GetSuitableFFTSize()` calculates the suitable FFT size.
/// The size is defined as the minimum length whose length is longer than
/// the input sample.
///
/// - Input
///     - `sample`  : Length of the input signal
///
/// - Output
///     - Suitable FFT size
pub fn GetSuitableFFTSize(sample: usize) -> usize {
    return (2.0_f64.powf((((sample as f64).ln() / world::kLog2) as i32 + 1) as f64)) as usize;
}

// These two functions are simple max() and min() function
// for types that implement PartialOrd.
#[inline]
pub fn MyMax<T: PartialOrd>(x: T, y: T) -> T {
    if x > y {
        x
    } else {
        y
    }
}

#[inline]
pub fn MyMin<T: PartialOrd>(x: T, y: T) -> T {
    if x < y {
        x
    } else {
        y
    }
}

// These functions are used in at least two different .cpp files

/// `DCCorrection()` interpolates the power under f0 Hz
/// and is used in `CheapTrick()` and `D4C()`.
pub fn DCCorrection(input: &mut [f64], f0: f64, fs: i32, fft_size: usize) {
    // current_f0 (on common.h) change to f0 (on common.cpp)
    let upper_limit = 2 + (f0 * fft_size as f64 / fs as f64) as usize;
    let mut low_frequency_replica: Vec<f64> = vec![0.0; upper_limit];
    let mut low_frequency_axis: Vec<f64> = vec![0.0; upper_limit];

    for i in 0..upper_limit {
        low_frequency_axis[i] = i as f64 * fs as f64 / fft_size as f64;
    }
    let upper_limit_replica = upper_limit - 1;
    interp1Q(
        f0 - low_frequency_axis[0],
        -fs as f64 / fft_size as f64,
        input,
        upper_limit + 1,
        &mut low_frequency_axis,
        upper_limit_replica,
        &mut low_frequency_replica,
    );

    for i in 0..upper_limit_replica {
        input[i] = input[i] + low_frequency_replica[i];
    }
}

/// `LinearSmoothing()` carries out the spectral smoothing by rectangular window
/// whose length is `width` Hz and is used in `CheapTrick()` and `D4C()`.
pub fn LinearSmoothing(
    input: Option<&[f64]>,
    width: f64,
    fs: i32,
    fft_size: usize,
    output: &mut [f64],
) {
    let input = input.unwrap_or(output.as_ref());
    let boundary = (width * fft_size as f64 / fs as f64) as usize + 1;

    // These parameters are set by the other function.
    let mut mirroring_spectrum: Vec<f64> = vec![0.0; fft_size / 2 + boundary * 2 + 1];
    let mut mirroring_segment: Vec<f64> = vec![0.0; fft_size / 2 + boundary * 2 + 1];
    let mut frequency_axis: Vec<f64> = vec![0.0; fft_size / 2 + 1];
    SetParametersForLinearSmoothing(
        boundary,
        fft_size,
        fs,
        width,
        input,
        &mut mirroring_spectrum,
        &mut mirroring_segment,
        &mut frequency_axis,
    );

    let mut low_levels: Vec<f64> = vec![0.0; fft_size / 2 + 1];
    let mut high_levels: Vec<f64> = vec![0.0; fft_size / 2 + 1];
    let origin_of_mirroring_axis: f64 = -(boundary as f64 - 0.5) * fs as f64 / fft_size as f64;
    let discrete_frequency_interval: f64 = fs as f64 / fft_size as f64;

    interp1Q(
        origin_of_mirroring_axis,
        discrete_frequency_interval,
        &mirroring_segment,
        fft_size / 2 + boundary * 2 + 1,
        &frequency_axis,
        fft_size / 2 + 1,
        &mut low_levels,
    );

    for i in 0..=fft_size / 2 {
        frequency_axis[i] += width;
    }

    interp1Q(
        origin_of_mirroring_axis,
        discrete_frequency_interval,
        &mirroring_segment,
        fft_size / 2 + boundary * 2 + 1,
        &frequency_axis,
        fft_size / 2 + 1,
        &mut high_levels,
    );

    for i in 0..=fft_size / 2 {
        output[i] = (high_levels[i] - low_levels[i]) / width;
    }
}

/// `NuttallWindow()` calculates the coefficients of Nuttall window whose length
/// is `y_length` and is used in `Dio()`, `Harvest()` and `D4C()`.
pub fn NuttallWindow(y_length: usize, y: &mut [f64]) {
    let mut tmp: f64;
    for i in 0..y_length {
        tmp = i as f64 / (y_length as f64 - 1.0);
        y[i] = 0.355768 - 0.487396 * (2.0 * world::kPi * tmp).cos()
            + 0.144232 * (4.0 * world::kPi * tmp).cos()
            - 0.012604 * (6.0 * world::kPi * tmp).cos();
    }
}

/// `GetSafeAperiodicity()` limit the range of aperiodicity from `0.001` to
/// `0.999999999999` (`1 - world::kMySafeGuardMinimum`).
#[inline]
pub fn GetSafeAperiodicity(x: f64) -> f64 {
    return MyMax(0.001, MyMin(0.999999999999, x));
}

// FFT, IFFT and minimum phase analysis
// These functions are used to speed up the processing.

/// For structs on FFT to FFT analysis
pub trait FFTProsess {
    fn new(fft_size: usize) -> Self;
    fn exec(&mut self);
}

// Forward FFT
impl FFTProsess for ForwardRealFFT<'_> {
    fn new(fft_size: usize) -> Self {
        Self {
            fft_size,
            waveform: vec![0.0; fft_size],
            spectrum: vec![fft_complex::default(); fft_size],
            forward_fft: fft_plan::default(),
        }
    }
    fn exec(&mut self) {
        let mut forward_fft = fft_plan_dft_r2c_1d(
            self.fft_size,
            Some(&self.waveform),
            Some(&mut self.spectrum),
            FFT_ESTIMATE,
        );
        fft_execute(&mut forward_fft);
    }
}

// Inverse FFT
impl FFTProsess for InverseRealFFT<'_> {
    fn new(fft_size: usize) -> Self {
        Self {
            fft_size: fft_size,
            spectrum: vec![fft_complex::default(); fft_size],
            waveform: vec![0.0; fft_size],
            inverse_fft: fft_plan::default(),
        }
    }
    fn exec(&mut self) {
        let mut inverse_fft = fft_plan_dft_c2r_1d(
            self.fft_size,
            Some(&self.spectrum),
            Some(&mut self.waveform),
            FFT_ESTIMATE,
        );
        fft_execute(&mut inverse_fft);
    }
}

// Inverse FFT (Complex)
impl FFTProsess for InverseComplexFFT<'_> {
    fn new(fft_size: usize) -> Self {
        Self {
            fft_size,
            input: vec![fft_complex::default(); fft_size],
            output: vec![fft_complex::default(); fft_size],
            inverse_fft: fft_plan::default(),
        }
    }
    fn exec(&mut self) {
        let mut inverse_fft = fft_plan_dft_1d(
            self.fft_size,
            Some(&self.input),
            Some(&mut self.output),
            FFT_BACKWARD,
            FFT_ESTIMATE,
        );
        fft_execute(&mut inverse_fft);
    }
}

// Minimum phase analysis (This analysis uses FFT)
impl MinimumPhaseAnalysis<'_> {
    pub fn new(fft_size: usize) -> Self {
        Self {
            fft_size,
            log_spectrum: vec![0.0; fft_size],
            cepstrum: vec![fft_complex::default(); fft_size],
            minimum_phase_spectrum: vec![fft_complex::default(); fft_size],
            inverse_fft: fft_plan::default(),
            forward_fft: fft_plan::default(),
        }
    }
    pub fn get_minimum_phase_spectrum(&mut self) {
        // Mirroring
        for i in self.fft_size / 2 + 1..self.fft_size {
            self.log_spectrum[i] = self.log_spectrum[self.fft_size - i];
        }

        // This fft_plan carries out "forward" FFT.
        // To carriy out the Inverse FFT, the sign of imaginary part
        // is inverted after FFT.
        self.exec_inverse_fft();
        self.cepstrum[0][1] *= -1.0;
        for i in 1..self.fft_size / 2 {
            self.cepstrum[i][0] *= 2.0;
            self.cepstrum[i][1] *= -2.0;
        }
        self.cepstrum[self.fft_size / 2][1] *= -1.0;
        for i in self.fft_size / 2 + 1..self.fft_size {
            self.cepstrum[i][0] = 0.0;
            self.cepstrum[i][1] = 0.0;
        }

        self.exec_forward_fft();

        // Since x is complex number, calculation of exp(x) is as following.
        // Note: This FFT library does not keep the aliasing.
        let mut tmp: f64;
        for i in 0..=self.fft_size / 2 {
            tmp = (self.minimum_phase_spectrum[i][0] / self.fft_size as f64).exp();
            self.minimum_phase_spectrum[i][0] =
                tmp * (self.minimum_phase_spectrum[i][1] / self.fft_size as f64).cos();
            self.minimum_phase_spectrum[i][1] =
                tmp * (self.minimum_phase_spectrum[i][1] / self.fft_size as f64).sin();
        }
    }
    fn exec_inverse_fft(&mut self) {
        let mut inverse_fft = fft_plan_dft_r2c_1d(
            self.fft_size,
            Some(&self.log_spectrum),
            Some(&mut self.cepstrum),
            FFT_ESTIMATE,
        );
        fft_execute(&mut inverse_fft);
    }
    fn exec_forward_fft(&mut self) {
        let mut forward_fft = fft_plan_dft_1d(
            self.fft_size,
            Some(&self.cepstrum),
            Some(&mut self.minimum_phase_spectrum),
            FFT_FORWARD,
            FFT_ESTIMATE,
        );
        fft_execute(&mut forward_fft);
    }
}
