//! Band-aperiodicity estimation on the basis of the idea of D4C.

use crate::common::{
    DCCorrection, FFTProsess, ForwardRealFFT, LinearSmoothing, MyMax, MyMin, NuttallWindow,
};
use crate::constantnumbers::world;
use crate::matlabfunctions::{interp1, matlab_round, Rng};

/// `SetParametersForGetWindowedWaveform()`
fn SetParametersForGetWindowedWaveform(
    half_window_length: usize,
    x_length: usize,
    current_position: f64,
    fs: i32,
    current_f0: f64,
    window_type: i32,
    window_length_ratio: f64,
    base_index: &mut [isize],
    safe_index: &mut [usize],
    window: &mut [f64],
) {
    for i in -(half_window_length as isize)..=half_window_length as isize {
        base_index[(i + half_window_length as isize) as usize] = i;
    }
    let origin = matlab_round(current_position * fs as f64 + 0.001) as isize;
    for i in 0..=half_window_length * 2 {
        safe_index[i] = MyMin(x_length - 1, MyMax(0, origin + base_index[i]) as usize);
    }

    // Designing of the window function
    let mut position: f64;
    if window_type == world::kHanning {
        // Hanning window
        for i in 0..=half_window_length * 2 {
            position = (2.0 * base_index[i] as f64 / window_length_ratio) / fs as f64;
            window[i] = 0.5 * (world::kPi * position * current_f0).cos() + 0.5;
        }
    } else {
        // Blackman window
        for i in 0..=half_window_length * 2 {
            position = (2.0 * base_index[i] as f64 / window_length_ratio) / fs as f64;
            window[i] = 0.42
                + 0.5 * (world::kPi * position * current_f0).cos()
                + 0.08 * (world::kPi * position * current_f0 * 2.0).cos();
        }
    }
}

/// `GetWindowedWaveform()` windows the waveform by F0-adaptive window
/// In the variable window_type, 1: hanning, 2: blackman
fn GetWindowedWaveform<R: Rng>(
    x: &[f64],
    x_length: usize,
    fs: i32,
    current_f0: f64,
    current_position: f64,
    window_type: i32,
    window_length_ratio: f64,
    waveform: &mut [f64],
    randn: &mut R,
) {
    let half_window_length =
        matlab_round(window_length_ratio * fs as f64 / current_f0 / 2.0) as usize;

    let mut base_index = vec![0; half_window_length * 2 + 1];
    let mut safe_index = vec![0; half_window_length * 2 + 1];
    let mut window = vec![0.0; half_window_length * 2 + 1];

    SetParametersForGetWindowedWaveform(
        half_window_length,
        x_length,
        current_position,
        fs,
        current_f0,
        window_type,
        window_length_ratio,
        &mut base_index,
        &mut safe_index,
        &mut window,
    );

    // F0-adaptive windowing
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

/// `GetCentroid()` calculates the energy centroid (see the book, time-frequency
/// analysis written by L. Cohen).
fn GetCentroid<R: Rng>(
    x: &[f64],
    x_length: usize,
    fs: i32,
    current_f0: f64,
    fft_size: usize,
    current_position: f64,
    forward_real_fft: &mut ForwardRealFFT,
    centroid: &mut [f64],
    randn: &mut R,
) {
    for i in 0..fft_size {
        forward_real_fft.waveform[i] = 0.0;
    }
    GetWindowedWaveform(
        x,
        x_length,
        fs,
        current_f0,
        current_position,
        world::kBlackman,
        4.0,
        &mut forward_real_fft.waveform,
        randn,
    );
    let mut power = 0.0;
    for i in 0..=matlab_round(2.0 * fs as f64 / current_f0) as usize * 2 {
        power += forward_real_fft.waveform[i] * forward_real_fft.waveform[i];
    }
    for i in 0..=matlab_round(2.0 * fs as f64 / current_f0) as usize * 2 {
        forward_real_fft.waveform[i] /= power.sqrt();
    }

    forward_real_fft.exec();
    let mut tmp_real = vec![0.0; fft_size / 2 + 1];
    let mut tmp_imag = vec![0.0; fft_size / 2 + 1];
    for i in 0..=fft_size / 2 {
        tmp_real[i] = forward_real_fft.spectrum[i][0];
        tmp_imag[i] = forward_real_fft.spectrum[i][1];
    }

    for i in 0..fft_size {
        forward_real_fft.waveform[i] *= i as f64 + 1.0;
    }
    forward_real_fft.exec();
    for i in 0..=fft_size / 2 {
        centroid[i] = forward_real_fft.spectrum[i][0] * tmp_real[i]
            + tmp_imag[i] * forward_real_fft.spectrum[i][1];
    }
}

/// `GetStaticCentroid()` calculates the temporally static energy centroid.
/// Basic idea was proposed by H. Kawahara.
fn GetStaticCentroid<R: Rng>(
    x: &[f64],
    x_length: usize,
    fs: i32,
    current_f0: f64,
    fft_size: usize,
    current_position: f64,
    forward_real_fft: &mut ForwardRealFFT,
    static_centroid: &mut [f64],
    randn: &mut R,
) {
    let mut centroid1 = vec![0.0; fft_size / 2 + 1];
    let mut centroid2 = vec![0.0; fft_size / 2 + 1];

    GetCentroid(
        x,
        x_length,
        fs,
        current_f0,
        fft_size,
        current_position - 0.25 / current_f0,
        forward_real_fft,
        &mut centroid1,
        randn,
    );
    GetCentroid(
        x,
        x_length,
        fs,
        current_f0,
        fft_size,
        current_position + 0.25 / current_f0,
        forward_real_fft,
        &mut centroid2,
        randn,
    );

    for i in 0..=fft_size / 2 {
        static_centroid[i] = centroid1[i] + centroid2[i];
    }

    DCCorrection(static_centroid, current_f0, fs, fft_size);
}

/// `GetSmoothedPowerSpectrum()` calculates the smoothed power spectrum.
/// The parameters used for smoothing are optimized in davance.
fn GetSmoothedPowerSpectrum<R: Rng>(
    x: &[f64],
    x_length: usize,
    fs: i32,
    current_f0: f64,
    fft_size: usize,
    current_position: f64,
    forward_real_fft: &mut ForwardRealFFT,
    smoothed_power_spectrum: &mut [f64],
    randn: &mut R,
) {
    for i in 0..fft_size {
        forward_real_fft.waveform[i] = 0.0;
    }
    GetWindowedWaveform(
        x,
        x_length,
        fs,
        current_f0,
        current_position,
        world::kHanning,
        4.0,
        &mut forward_real_fft.waveform,
        randn,
    );

    //fft_execute(forward_real_fft.forward_fft);
    forward_real_fft.exec();
    for i in 0..=fft_size / 2 {
        smoothed_power_spectrum[i] = forward_real_fft.spectrum[i][0]
            * forward_real_fft.spectrum[i][0]
            + forward_real_fft.spectrum[i][1] * forward_real_fft.spectrum[i][1];
    }
    DCCorrection(smoothed_power_spectrum, current_f0, fs, fft_size);
    LinearSmoothing(None, current_f0, fs, fft_size, smoothed_power_spectrum);
}

/// `GetStaticGroupDelay()` calculates the temporally static group delay.
/// This is the fundamental parameter in D4C.
fn GetStaticGroupDelay(
    static_centroid: &[f64],
    smoothed_power_spectrum: &[f64],
    fs: i32,
    f0: f64,
    fft_size: usize,
    static_group_delay: &mut [f64],
) {
    for i in 0..=fft_size / 2 {
        static_group_delay[i] = static_centroid[i] / smoothed_power_spectrum[i];
    }
    LinearSmoothing(None, f0 / 2.0, fs, fft_size, static_group_delay);

    let mut smoothed_group_delay = vec![0.0; fft_size / 2 + 1];
    LinearSmoothing(
        Some(static_group_delay),
        f0,
        fs,
        fft_size,
        &mut smoothed_group_delay,
    );

    for i in 0..=fft_size / 2 {
        static_group_delay[i] -= smoothed_group_delay[i];
    }
}

/// `GetCoarseAperiodicity()` calculates the aperiodicity in multiples of 3 kHz.
/// The upper limit is given based on the sampling frequency.
fn GetCoarseAperiodicity(
    static_group_delay: &[f64],
    fs: i32,
    fft_size: usize,
    number_of_aperiodicities: usize,
    window: &[f64],
    window_length: usize,
    forward_real_fft: &mut ForwardRealFFT,
    coarse_aperiodicity: &mut [f64],
) {
    let boundary = matlab_round(fft_size as f64 * 8.0 / window_length as f64) as usize;
    let half_window_length = window_length / 2;

    for i in 0..fft_size {
        forward_real_fft.waveform[i] = 0.0;
    }

    let mut power_spectrum = vec![0.0; fft_size / 2 + 1];
    let mut center: usize;
    for i in 0..number_of_aperiodicities {
        center =
            (world::kFrequencyInterval * (i + 1) as f64 * fft_size as f64 / fs as f64) as usize;
        for j in 0..=half_window_length * 2 {
            forward_real_fft.waveform[j] =
                static_group_delay[center - half_window_length + j] * window[j];
        }
        forward_real_fft.exec();
        for j in 0..=fft_size / 2 {
            power_spectrum[j] = forward_real_fft.spectrum[j][0] * forward_real_fft.spectrum[j][0]
                + forward_real_fft.spectrum[j][1] * forward_real_fft.spectrum[j][1];
        }
        power_spectrum.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for j in 1..=fft_size / 2 {
            power_spectrum[j] += power_spectrum[j - 1];
        }
        coarse_aperiodicity[i] = 10.0
            * (power_spectrum[fft_size / 2 - boundary - 1] / power_spectrum[fft_size / 2]).log10();
    }
}

fn D4CLoveTrainSub<R: Rng>(
    x: &[f64],
    fs: i32,
    x_length: usize,
    current_f0: f64,
    current_position: f64,
    _f0_length: usize,
    fft_size: usize,
    boundary0: usize,
    boundary1: usize,
    boundary2: usize,
    forward_real_fft: &mut ForwardRealFFT,
    randn: &mut R,
) -> f64 {
    let mut power_spectrum = vec![0.0; fft_size];

    let window_length = matlab_round(1.5 * fs as f64 / current_f0) as usize * 2 + 1;
    GetWindowedWaveform(
        x,
        x_length,
        fs,
        current_f0,
        current_position,
        world::kBlackman,
        3.0,
        &mut forward_real_fft.waveform,
        randn,
    );

    for i in window_length..fft_size {
        forward_real_fft.waveform[i] = 0.0;
    }
    forward_real_fft.exec();

    for i in 0..=boundary0 {
        power_spectrum[i] = 0.0;
    }
    for i in boundary0 + 1..fft_size / 2 + 1 {
        power_spectrum[i] = forward_real_fft.spectrum[i][0] * forward_real_fft.spectrum[i][0]
            + forward_real_fft.spectrum[i][1] * forward_real_fft.spectrum[i][1];
    }
    for i in boundary0..=boundary2 {
        power_spectrum[i] += power_spectrum[i - 1];
    }

    let aperiodicity0 = power_spectrum[boundary1] / power_spectrum[boundary2];
    return aperiodicity0;
}

/// `D4CLoveTrain()` determines the aperiodicity with VUV detection.
/// If a frame was determined as the unvoiced section, aperiodicity is set to
/// very high value as the safeguard.
/// If it was voiced section, the aperiodicity of 0 Hz is set to -60 dB.
fn D4CLoveTrain<R: Rng>(
    x: &[f64],
    fs: i32,
    x_length: usize,
    f0: &[f64],
    f0_length: usize,
    temporal_positions: &[f64],
    aperiodicity0: &mut [f64],
    randn: &mut R,
) {
    let lowest_f0 = 40.0;
    let fft_size = 2.0_f64
        .powf(1.0 + ((3.0 * fs as f64 / lowest_f0 + 1.0).ln() / world::kLog2).trunc())
        as usize;
    let mut forward_real_fft = ForwardRealFFT::new(fft_size);

    // Cumulative powers at 100, 4000, 7900 Hz are used for VUV identification.
    let boundary0 = (100.0 * fft_size as f64 / fs as f64).ceil() as usize;
    let boundary1 = (4000.0 * fft_size as f64 / fs as f64).ceil() as usize;
    let boundary2 = (7900.0 * fft_size as f64 / fs as f64).ceil() as usize;
    for i in 0..f0_length {
        if f0[i] == 0.0 {
            aperiodicity0[i] = 0.0;
            continue;
        }
        aperiodicity0[i] = D4CLoveTrainSub(
            x,
            fs,
            x_length,
            MyMax(f0[i], lowest_f0),
            temporal_positions[i],
            f0_length,
            fft_size,
            boundary0,
            boundary1,
            boundary2,
            &mut forward_real_fft,
            randn,
        );
    }
}

/// `D4CGeneralBody()` calculates a spectral envelope at a temporal
/// position. This function is only used in `D4C()`.
/// Caution:
///     forward_fft is allocated in advance to speed up the processing.
fn D4CGeneralBody<R: Rng>(
    x: &[f64],
    x_length: usize,
    fs: i32,
    current_f0: f64,
    fft_size: usize,
    current_position: f64,
    number_of_aperiodicities: usize,
    window: &[f64],
    window_length: usize,
    forward_real_fft: &mut ForwardRealFFT,
    coarse_aperiodicity: &mut [f64],
    randn: &mut R,
) {
    let mut static_centroid = vec![0.0; fft_size / 2 + 1];
    let mut smoothed_power_spectrum = vec![0.0; fft_size / 2 + 1];
    let mut static_group_delay = vec![0.0; fft_size / 2 + 1];
    GetStaticCentroid(
        x,
        x_length,
        fs,
        current_f0,
        fft_size,
        current_position,
        forward_real_fft,
        &mut static_centroid,
        randn,
    );
    GetSmoothedPowerSpectrum(
        x,
        x_length,
        fs,
        current_f0,
        fft_size,
        current_position,
        forward_real_fft,
        &mut smoothed_power_spectrum,
        randn,
    );
    GetStaticGroupDelay(
        &static_centroid,
        &smoothed_power_spectrum,
        fs,
        current_f0,
        fft_size,
        &mut static_group_delay,
    );

    GetCoarseAperiodicity(
        &static_group_delay,
        fs,
        fft_size,
        number_of_aperiodicities,
        window,
        window_length,
        forward_real_fft,
        coarse_aperiodicity,
    );

    // Revision of the result based on the F0
    for i in 0..number_of_aperiodicities {
        coarse_aperiodicity[i] = MyMin(0.0, coarse_aperiodicity[i] + (current_f0 - 100.0) / 50.0);
    }
}

fn InitializeAperiodicity(f0_length: usize, fft_size: usize, aperiodicity: &mut [Vec<f64>]) {
    for i in 0..f0_length {
        for j in 0..fft_size / 2 + 1 {
            aperiodicity[i][j] = 1.0 - world::kMySafeGuardMinimum;
        }
    }
}

fn GetAperiodicity(
    coarse_frequency_axis: &[f64],
    coarse_aperiodicity: &[f64],
    number_of_aperiodicities: usize,
    frequency_axis: &[f64],
    fft_size: usize,
    aperiodicity: &mut [f64],
) {
    interp1(
        coarse_frequency_axis,
        coarse_aperiodicity,
        number_of_aperiodicities + 2,
        frequency_axis,
        fft_size / 2 + 1,
        aperiodicity,
    );
    for i in 0..=fft_size / 2 {
        aperiodicity[i] = 10.0_f64.powf(aperiodicity[i] / 20.0);
    }
}

//---

/// Struct for D4C
pub struct D4COption {
    threshold: f64,
}

/// `D4C()` calculates the aperiodicity estimated by D4C.
///
/// - Input
///     - `x`                   : Input signal
///     - `x_length`            : Length of `x`
///     - `fs`                  : Sampling frequency
///     - `temporal_positions`  : Time axis
///     - `f0`                  : F0 contour
///     - `f0_length`           : Length of F0 contour
///     - `fft_size`            : Number of samples of the aperiodicity in one frame.
///     It is given by the equation `fft_size / 2 + 1`.
/// - Output
///   aperiodicity  : Aperiodicity estimated by D4C.
pub fn D4C<R: Rng>(
    x: &[f64],
    x_length: usize,
    fs: i32,
    temporal_positions: &[f64],
    f0: &[f64],
    f0_length: usize,
    fft_size: usize,
    option: &D4COption,
    aperiodicity: &mut [Vec<f64>],
) {
    let mut randn = R::new();

    InitializeAperiodicity(f0_length, fft_size, aperiodicity);

    let fft_size_d4c = 2.0_f64
        .powf(1.0 + ((4.0 * fs as f64 / world::kFloorF0D4C + 1.0).ln() / world::kLog2).trunc())
        as usize;
    let mut forward_real_fft = ForwardRealFFT::new(fft_size_d4c);

    let number_of_aperiodicities = (MyMin(
        world::kUpperLimit,
        fs as f64 / 2.0 - world::kFrequencyInterval,
    ) / world::kFrequencyInterval) as usize;
    // Since the window function is common in `D4CGeneralBody()`,
    // it is designed here to speed up.
    let window_length =
        (world::kFrequencyInterval * fft_size_d4c as f64 / fs as f64) as usize * 2 + 1;
    let mut window = vec![0.0; window_length];
    NuttallWindow(window_length, &mut window);

    // D4C Love Train (Aperiodicity of 0 Hz is given by the different algorithm)
    let mut aperiodicity0 = vec![0.0; f0_length];
    D4CLoveTrain(
        x,
        fs,
        x_length,
        f0,
        f0_length,
        temporal_positions,
        &mut aperiodicity0,
        &mut randn,
    );

    let mut coarse_aperiodicity = vec![0.0; number_of_aperiodicities + 2];
    coarse_aperiodicity[0] = -60.0;
    coarse_aperiodicity[number_of_aperiodicities + 1] = -world::kMySafeGuardMinimum;
    let mut coarse_frequency_axis = vec![0.0; number_of_aperiodicities + 2];
    for i in 0..=number_of_aperiodicities {
        coarse_frequency_axis[i] = i as f64 * world::kFrequencyInterval;
    }
    coarse_frequency_axis[number_of_aperiodicities + 1] = fs as f64 / 2.0;

    let mut frequency_axis = vec![0.0; fft_size / 2 + 1];
    for i in 0..=fft_size / 2 {
        frequency_axis[i] = i as f64 * fs as f64 / fft_size as f64;
    }

    for i in 0..f0_length {
        if f0[i] == 0.0 || aperiodicity0[i] <= option.threshold {
            continue;
        }
        D4CGeneralBody(
            x,
            x_length,
            fs,
            MyMax(world::kFloorF0D4C, f0[i]),
            fft_size_d4c,
            temporal_positions[i],
            number_of_aperiodicities,
            &window,
            window_length,
            &mut forward_real_fft,
            &mut coarse_aperiodicity[1..],
            &mut randn,
        );
        // Linear interpolation to convert the coarse aperiodicity into its
        // spectral representation.
        GetAperiodicity(
            &coarse_frequency_axis,
            &coarse_aperiodicity,
            number_of_aperiodicities,
            &frequency_axis,
            fft_size,
            &mut aperiodicity[i],
        );
    }
}

impl D4COption {
    /// Set the default parameters.
    ///
    /// - Output
    ///     - Struct for the optional parameter.
    pub fn new() -> Self {
        Self {
            threshold: world::kThreshold,
        }
    }

    pub fn threshold(self, threshold: f64) -> Self {
        Self { threshold }
    }
}
