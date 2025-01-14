//! Spectral envelope estimation on the basis of the idea of CheapTrick.

use crate::common::{
    DCCorrection, FFTProsess, ForwardRealFFT, InverseRealFFT, LinearSmoothing, MyMax, MyMin,
};
use crate::constantnumbers::world;
use crate::matlabfunctions::{matlab_round, Rng};

/// `SmoothingWithRecovery()` carries out the spectral smoothing and spectral
/// recovery on the Cepstrum domain.
fn SmoothingWithRecovery(
    f0: f64,
    fs: i32,
    fft_size: usize,
    q1: f64,
    forward_real_fft: &mut ForwardRealFFT,
    inverse_real_fft: &mut InverseRealFFT,
    spectral_envelope: &mut [f64],
) {
    let mut smoothing_lifter = vec![0.0; fft_size];
    let mut compensation_lifter: Vec<f64> = vec![0.0; fft_size];

    smoothing_lifter[0] = 1.0;
    compensation_lifter[0] = (1.0 - 2.0 * q1) + 2.0 * q1;
    let mut quefrency: f64;
    for i in 1..=forward_real_fft.fft_size / 2 {
        quefrency = i as f64 / fs as f64;
        smoothing_lifter[i] = (world::kPi * f0 * quefrency).sin() / (world::kPi * f0 * quefrency);
        compensation_lifter[i] =
            (1.0 - 2.0 * q1) + 2.0 * q1 * (2.0 * world::kPi * quefrency * f0).cos();
    }

    for i in 0..=fft_size / 2 {
        forward_real_fft.waveform[i] = forward_real_fft.waveform[i].ln();
    }
    for i in 1..fft_size / 2 {
        forward_real_fft.waveform[fft_size - i] = forward_real_fft.waveform[i];
    }
    forward_real_fft.exec();

    for i in 0..=fft_size / 2 {
        inverse_real_fft.spectrum[i][0] =
            forward_real_fft.spectrum[i][0] * smoothing_lifter[i] * compensation_lifter[i]
                / fft_size as f64;
        inverse_real_fft.spectrum[i][1] = 0.0;
    }
    inverse_real_fft.exec();

    for i in 0..=fft_size / 2 {
        spectral_envelope[i] = inverse_real_fft.waveform[i].exp();
    }
}

/// `GetPowerSpectrum()` calculates the power_spectrum with DC correction.
/// DC stands for Direct Current. In this case, the component from 0 to F0 Hz
/// is corrected.
fn GetPowerSpectrum(fs: i32, f0: f64, fft_size: usize, forward_real_fft: &mut ForwardRealFFT) {
    let half_window_length = matlab_round(1.5 * fs as f64 / f0) as usize;

    // FFT
    for i in half_window_length * 2 + 1..fft_size {
        forward_real_fft.waveform[i] = 0.0;
    }
    //fft_execute(forward_real_fft.forward_fft);
    forward_real_fft.exec();

    // Calculation of the power spectrum.
    let power_spectrum = &mut forward_real_fft.waveform;
    for i in 0..=fft_size / 2 {
        power_spectrum[i] = forward_real_fft.spectrum[i][0] * forward_real_fft.spectrum[i][0]
            + forward_real_fft.spectrum[i][1] * forward_real_fft.spectrum[i][1];
    }

    // DC correction
    DCCorrection(power_spectrum, f0, fs, fft_size);
}

/// `SetParametersForGetWindowedWaveform()`
fn SetParametersForGetWindowedWaveform(
    half_window_length: usize,
    x_length: usize,
    currnet_position: f64,
    fs: i32,
    current_f0: f64,
    base_index: &mut [isize],
    safe_index: &mut [usize],
    window: &mut [f64],
) {
    for i in -(half_window_length as isize)..=half_window_length as isize {
        base_index[(i + half_window_length as isize) as usize] = i;
    }
    let origin = matlab_round(currnet_position * fs as f64 + 0.001) as isize;
    for i in 0..=half_window_length * 2 {
        safe_index[i] = MyMin(x_length - 1, MyMax(0, origin + base_index[i]) as usize);
    }

    // Designing of the window function
    let mut average = 0.0;
    let mut position: f64;
    for i in 0..=half_window_length * 2 {
        position = base_index[i] as f64 / 1.5 / fs as f64;
        window[i] = 0.5 * (world::kPi * position * current_f0).cos() + 0.5;
        average += window[i] * window[i];
    }
    average = average.sqrt();
    for i in 0..=half_window_length * 2 {
        window[i] /= average;
    }
}

/// `GetWindowedWaveform()` windows the waveform by F0-adaptive window
fn GetWindowedWaveform<R: Rng>(
    x: &[f64],
    x_length: usize,
    fs: i32,
    current_f0: f64,
    currnet_position: f64,
    forward_real_fft: &mut ForwardRealFFT,
    randn: &mut R,
) {
    let half_window_length = matlab_round(1.5 * fs as f64 / current_f0) as usize;

    let mut base_index = vec![0; half_window_length * 2 + 1];
    let mut safe_index = vec![0; half_window_length * 2 + 1];
    let mut window = vec![0.0; half_window_length * 2 + 1];

    SetParametersForGetWindowedWaveform(
        half_window_length,
        x_length,
        currnet_position,
        fs,
        current_f0,
        &mut base_index,
        &mut safe_index,
        &mut window,
    );

    // F0-adaptive windowing
    let waveform = &mut forward_real_fft.waveform;
    for i in 0..=half_window_length * 2 {
        waveform[i] = x[safe_index[i]] * window[i] + randn.gen() * world::kMySafeGuardMinimum;
    }
    let mut tmp_weight1 = 0.0;
    let mut tmp_weight2 = 0.0;
    for i in 0..=half_window_length * 2 {
        tmp_weight1 += waveform[i];
        tmp_weight2 += window[i];
    }
    let weighting_coefficient = tmp_weight1 / tmp_weight2;
    for i in 0..=half_window_length * 2 {
        waveform[i] -= window[i] * weighting_coefficient;
    }
}

/// `AddInfinitesimalNoise()`
fn AddInfinitesimalNoise<R: Rng>(input_spectrum: &mut [f64], fft_size: usize, randn: &mut R) {
    for i in 0..=fft_size / 2 {
        input_spectrum[i] = input_spectrum[i] + randn.gen().abs() * world::kEps;
    }
}

/// `CheapTrickGeneralBody()` calculates a spectral envelope at a temporal
/// position. This function is only used in `CheapTrick()`.
/// Caution:
///     `forward_fft` is allocated in advance to speed up the processing.
fn CheapTrickGeneralBody<R: Rng>(
    x: &[f64],
    x_length: usize,
    fs: i32,
    current_f0: f64,
    fft_size: usize,
    current_position: f64,
    q1: f64,
    forward_real_fft: &mut ForwardRealFFT,
    inverse_real_fft: &mut InverseRealFFT,
    spectral_envelope: &mut [f64],
    randn: &mut R,
) {
    // F0-adaptive windowing
    GetWindowedWaveform(
        x,
        x_length,
        fs,
        current_f0,
        current_position,
        forward_real_fft,
        randn,
    );

    // Calculate power spectrum with DC correction
    // Note: The calculated power spectrum is stored in an array for waveform.
    // In this imprementation, power spectrum is transformed by FFT (NOT IFFT).
    // However, the same result is obtained.
    // This is tricky but important for simple implementation.
    GetPowerSpectrum(fs, current_f0, fft_size, forward_real_fft);

    // Smoothing of the power (linear axis)
    // forward_real_fft.waveform is the power spectrum.
    LinearSmoothing(
        None,
        current_f0 * 2.0 / 3.0,
        fs,
        fft_size,
        &mut forward_real_fft.waveform,
    );

    // Add infinitesimal noise
    // This is a safeguard to avoid including zero in the spectrum.
    AddInfinitesimalNoise(&mut forward_real_fft.waveform, fft_size, randn);

    // Smoothing (log axis) and spectral recovery on the cepstrum domain.
    SmoothingWithRecovery(
        current_f0,
        fs,
        fft_size,
        q1,
        forward_real_fft,
        inverse_real_fft,
        spectral_envelope,
    );
}

//---

/// Struct for CheapTrick
pub struct CheapTrickOption {
    q1: f64,
    f0_floor: f64,
    fft_size: usize,
}

/// `CheapTrick()` calculates the spectrogram that consists of spectral envelopes
/// estimated by CheapTrick.
///
/// - Input
///     - `x`                   : Input signal
///     - `x_length`            : Length of `x`
///     - `fs`                  : Sampling frequency
///     - `temporal_positions`  : Time axis
///     - `f0`                  : F0 contour
///     - `f0_length`           : Length of F0 contour
///     - `option`              : Struct to order the parameter for CheapTrick
///
/// - Output
///     - `spectrogram`         : Spectrogram estimated by CheapTrick.
pub fn CheapTrick<R: Rng>(
    x: &[f64],
    x_length: usize,
    fs: i32,
    temporal_positions: &[f64],
    f0: &[f64],
    f0_length: usize,
    option: &CheapTrickOption,
    spectrogram: &mut [Vec<f64>],
) {
    let fft_size = option.fft_size;

    let mut randn = R::new();

    let f0_floor = GetF0FloorForCheapTrick(fs, fft_size);
    let mut spectral_envelope = vec![0.0; fft_size];

    let mut forward_real_fft = ForwardRealFFT::new(fft_size);
    let mut inverse_real_fft = InverseRealFFT::new(fft_size);
    let mut current_f0: f64;
    for i in 0..f0_length {
        current_f0 = if f0[i] <= f0_floor {
            world::kDefaultF0
        } else {
            f0[i]
        };
        CheapTrickGeneralBody(
            x,
            x_length,
            fs,
            current_f0,
            fft_size,
            temporal_positions[i],
            option.q1,
            &mut forward_real_fft,
            &mut inverse_real_fft,
            &mut spectral_envelope,
            &mut randn,
        );
        for j in 0..=fft_size / 2 {
            spectrogram[i][j] = spectral_envelope[j];
        }
    }
}

impl CheapTrickOption {
    /// Set the default parameters.
    ///
    /// - Input
    ///     - `fs`      : Sampling frequency
    ///
    /// - Output
    ///     - Struct for the optional parameter
    pub fn new(fs: i32) -> Self {
        let f0_floor = world::kFloorF0;
        let fft_size = Self::get_fftsize_for_cheap_trick(f0_floor, fs);
        Self {
            q1: -0.15,
            f0_floor,
            fft_size,
        }
    }

    /// `q1` is the parameter used for the spectral recovery.
    /// Since The parameter is optimized, you don't need to change the parameter.
    pub fn q1(self, q1: f64) -> Self {
        Self { q1, ..self }
    }

    /// `f0_floor` and `fs` are used to determine `fft_size`;
    /// We strongly recommend not to change this value unless you have enough
    /// knowledge of the signal processing in CheapTrick.
    pub fn f0_floor(self, f0_floor: f64, fs: i32) -> Self {
        let fft_size = Self::get_fftsize_for_cheap_trick(f0_floor, fs);
        Self {
            f0_floor,
            fft_size,
            ..self
        }
    }

    /// Calculate the FFT size based on the sampling
    /// frequency and the lower limit of f0 (`kFloorF0` defined in `constantnumbers`).
    ///
    /// - Input
    ///     - `f0_floor`    : Lower limit of f0
    ///     - `fs`          : Sampling frequency
    ///
    /// - Output
    ///     - FFT size
    #[inline]
    fn get_fftsize_for_cheap_trick(f0_floor: f64, fs: i32) -> usize {
        return 2.0_f64.powf(1.0 + ((3.0 * fs as f64 / f0_floor + 1.0).ln() / world::kLog2).trunc())
            as usize;
    }

    /// Wrapper function of `CheapTrickOption::get_fftsize_for_cheap_trick()` for compatibility.
    ///
    /// - Input
    ///     - `fs`  : Sampling frequency
    ///
    /// - Output
    ///     - FFT size
    pub fn GetFFTSizeForCheapTrick(&self, fs: i32) -> usize {
        CheapTrickOption::get_fftsize_for_cheap_trick(self.f0_floor, fs)
    }
}

/// GetF0FloorForCheapTrick() calculates actual lower f0 limit for CheapTrick
/// based on the sampling frequency and FFT size used. Whenever f0 is below
/// this threshold the spectrum will be analyzed as if the frame is unvoiced
/// (using kDefaultF0 defined in constantnumbers.h).
///
/// - Input
///   fs : Sampling frequency
///   fft_size : FFT size
///
/// - Output
///   Lower f0 limit (Hz)
pub fn GetF0FloorForCheapTrick(fs: i32, fft_size: usize) -> f64 {
    return 3.0 * fs as f64 / (fft_size as f64 - 3.0);
}
