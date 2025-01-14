//! F0 estimation based on instantaneous frequency.
//! This method is carried out by using the output of `Dio()`.

use crate::common::{FFTProsess, ForwardRealFFT, MyMax, MyMin};
use crate::constantnumbers::world;
use crate::fft::fft_complex;
use crate::matlabfunctions::matlab_round;

//---

/// `GetBaseIndex()` calculates the temporal positions for windowing.
/// Since the result includes negative value and the value that exceeds the
/// length of the input signal, it must be modified appropriately.
fn GetBaseIndex(
    current_position: f64,
    base_time: &[f64],
    base_time_length: usize,
    fs: i32,
    index_raw: &mut [isize],
) {
    for i in 0..base_time_length {
        index_raw[i] = matlab_round((current_position + base_time[i]) * fs as f64) as isize;
    }
}

/// `GetMainWindow()` generates the window function.
fn GetMainWindow(
    current_position: f64,
    index_raw: &[isize],
    base_time_length: usize,
    fs: i32,
    window_length_in_time: f64,
    main_window: &mut [f64],
) {
    for i in 0..base_time_length {
        let tmp = (index_raw[i] as f64 - 1.0) / fs as f64 - current_position;
        main_window[i] = 0.42
            + 0.5 * (2.0 * world::kPi * tmp / window_length_in_time).cos()
            + 0.08 * (4.0 * world::kPi * tmp / window_length_in_time).cos();
    }
}

/// `GetDiffWindow()` generates the differentiated window.
/// Diff means differential.
fn GetDiffWindow(main_window: &[f64], base_time_length: usize, diff_window: &mut [f64]) {
    diff_window[0] = -main_window[1] / 2.0;
    for i in 1..base_time_length - 1 {
        diff_window[i] = -(main_window[i + 1] - main_window[i - 1]) / 2.0;
    }
    diff_window[base_time_length - 1] = main_window[base_time_length - 2] / 2.0;
}

/// `GetSpectra()` calculates two spectra of the waveform windowed by windows
/// (main window and diff window).
fn GetSpectra(
    x: &[f64],
    x_length: usize,
    fft_size: usize,
    index_raw: &[isize],
    main_window: &[f64],
    diff_window: &[f64],
    base_time_length: usize,
    forward_real_fft: &mut ForwardRealFFT,
    main_spectrum: &mut [fft_complex],
    diff_spectrum: &mut [fft_complex],
) {
    let mut index = vec![0; base_time_length];

    for i in 0..base_time_length {
        index[i] = MyMax(0, MyMin((x_length - 1) as isize, index_raw[i] - 1)) as usize;
    }
    for i in 0..base_time_length {
        forward_real_fft.waveform[i] = x[index[i]] * main_window[i];
    }
    for i in base_time_length..fft_size {
        forward_real_fft.waveform[i] = 0.0;
    }

    forward_real_fft.exec();
    for i in 0..=fft_size / 2 {
        main_spectrum[i][0] = forward_real_fft.spectrum[i][0];
        main_spectrum[i][1] = forward_real_fft.spectrum[i][1];
    }

    for i in 0..base_time_length {
        forward_real_fft.waveform[i] = x[index[i]] * diff_window[i];
    }
    for i in base_time_length..fft_size {
        forward_real_fft.waveform[i] = 0.0;
    }
    forward_real_fft.exec();
    for i in 0..=fft_size / 2 {
        diff_spectrum[i][0] = forward_real_fft.spectrum[i][0];
        diff_spectrum[i][1] = forward_real_fft.spectrum[i][1];
    }
}

/// `FixF0()` fixed the F0 by instantaneous frequency.
fn FixF0(
    power_spectrum: &[f64],
    numerator_i: &[f64],
    fft_size: usize,
    fs: i32,
    initial_f0: f64,
    number_of_harmonics: usize,
) -> f64 {
    let mut amplitude_list = vec![0.0; number_of_harmonics];
    let mut instantaneous_frequency_list = vec![0.0; number_of_harmonics];
    for i in 0..number_of_harmonics {
        let index = MyMin(
            matlab_round(initial_f0 * fft_size as f64 / fs as f64 * (i + 1) as f64) as usize,
            fft_size / 2,
        );
        instantaneous_frequency_list[i] = if power_spectrum[index] == 0.0 {
            0.0
        } else {
            index as f64 * fs as f64 / fft_size as f64
                + numerator_i[index] / power_spectrum[index] * fs as f64 / 2.0 / world::kPi
        };
        amplitude_list[i] = power_spectrum[index].sqrt();
    }
    let mut denominator = 0.0;
    let mut numerator = 0.0;
    for i in 0..number_of_harmonics {
        numerator += amplitude_list[i] * instantaneous_frequency_list[i];
        denominator += amplitude_list[i] * (i + 1) as f64;
    }
    return numerator / (denominator + world::kMySafeGuardMinimum);
}

/// `GetTentativeF0()` calculates the F0 based on the instantaneous frequency.
fn GetTentativeF0(
    power_spectrum: &[f64],
    numerator_i: &[f64],
    fft_size: usize,
    fs: i32,
    initial_f0: f64,
) -> f64 {
    let tentative_f0 = FixF0(power_spectrum, numerator_i, fft_size, fs, initial_f0, 2);

    // If the fixed value is too large, the result will be rejected.
    if tentative_f0 <= 0.0 || tentative_f0 > initial_f0 * 2.0 {
        0.0
    } else {
        FixF0(power_spectrum, numerator_i, fft_size, fs, tentative_f0, 6)
    }
}

/// `GetMeanF0()` calculates the instantaneous frequency.
fn GetMeanF0(
    x: &[f64],
    x_length: usize,
    fs: i32,
    current_position: f64,
    initial_f0: f64,
    fft_size: usize,
    window_length_in_time: f64,
    base_time: &[f64],
    base_time_length: usize,
) -> f64 {
    let mut forward_real_fft = ForwardRealFFT::new(fft_size);
    let mut main_spectrum = vec![fft_complex::default(); fft_size];
    let mut diff_spectrum = vec![fft_complex::default(); fft_size];

    let mut index_raw = vec![0; base_time_length];
    let mut main_window = vec![0.0; base_time_length];
    let mut diff_window = vec![0.0; base_time_length];

    GetBaseIndex(
        current_position,
        base_time,
        base_time_length,
        fs,
        &mut index_raw,
    );
    GetMainWindow(
        current_position,
        &index_raw,
        base_time_length,
        fs,
        window_length_in_time,
        &mut main_window,
    );
    GetDiffWindow(&main_window, base_time_length, &mut diff_window);
    GetSpectra(
        x,
        x_length,
        fft_size,
        &index_raw,
        &main_window,
        &diff_window,
        base_time_length,
        &mut forward_real_fft,
        &mut main_spectrum,
        &mut diff_spectrum,
    );

    let mut power_spectrum = vec![0.0; fft_size / 2 + 1];
    let mut numerator_i = vec![0.0; fft_size / 2 + 1];
    for j in 0..=fft_size / 2 {
        numerator_i[j] =
            main_spectrum[j][0] * diff_spectrum[j][1] - main_spectrum[j][1] * diff_spectrum[j][0];
        power_spectrum[j] =
            main_spectrum[j][0] * main_spectrum[j][0] + main_spectrum[j][1] * main_spectrum[j][1];
    }

    let tentative_f0 = GetTentativeF0(&power_spectrum, &numerator_i, fft_size, fs, initial_f0);

    return tentative_f0;
}

/// `GetRefinedF0()` fixes the F0 estimated by `Dio()`. This function uses
/// instantaneous frequency.
fn GetRefinedF0(
    x: &[f64],
    x_length: usize,
    fs: i32,
    current_potision: f64,
    initial_f0: f64,
) -> f64 {
    if initial_f0 <= world::kFloorF0StoneMask || initial_f0 > fs as f64 / 12.0 {
        return 0.0;
    }

    let half_window_length = (1.5 * fs as f64 / initial_f0 + 1.0) as usize;
    let window_length_in_time = (2.0 * half_window_length as f64 + 1.0) / fs as f64;
    let mut base_time = vec![0.0; half_window_length * 2 + 1];
    for i in 0..half_window_length * 2 + 1 {
        base_time[i] = (i as isize - half_window_length as isize) as f64 / fs as f64;
    }
    let fft_size = 2.0_f64
        .powf(2.0 + ((half_window_length as f64 * 2.0 + 1.0).ln() / world::kLog2).trunc())
        as usize;

    let mean_f0 = GetMeanF0(
        x,
        x_length,
        fs,
        current_potision,
        initial_f0,
        fft_size,
        window_length_in_time,
        &base_time,
        half_window_length * 2 + 1,
    );

    // If amount of correction is overlarge (20 %), initial F0 is employed.
    if (mean_f0 - initial_f0).abs() > initial_f0 * 0.2 {
        initial_f0
    } else {
        mean_f0
    }
}

//---

/// `StoneMask()` refines the estimated F0 by `Dio()`
///
/// - Input
///   - `x`             : Input signal
///   - `x_length`      : Length of the input signal
///   - `fs`            : Sampling frequency
///   - `time_axis`     : Temporal information
///   - `f0`            : f0 contour
///   - `f0_length`     : Length of `f0`
///
/// - Output
///   - `refined_f0`    : Refined F0
pub fn StoneMask(
    x: &[f64],
    x_length: usize,
    fs: i32,
    temporal_positions: &[f64],
    f0: &[f64],
    f0_length: usize,
    refined_f0: &mut [f64],
) {
    for i in 0..f0_length {
        refined_f0[i] = GetRefinedF0(x, x_length, fs, temporal_positions[i], f0[i]);
    }
}
