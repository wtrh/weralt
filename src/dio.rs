//! F0 estimation based on DIO (Distributed Inline-filter Operation).

use crate::common::{GetSuitableFFTSize, MyMax, MyMin, NuttallWindow};
use crate::constantnumbers::world;
use crate::fft::{
    fft_complex, fft_execute, fft_plan_dft_c2r_1d, fft_plan_dft_r2c_1d, FFT_ESTIMATE,
};
use crate::matlabfunctions::{decimate, interp1, matlab_round};

/// struct for `GetFourZeroCrossingIntervals()`
/// - "negative" means "zero-crossing point going from positive to negative"
/// - "positive" means "zero-crossing point going from negative to positive"
#[derive(Default)]
struct ZeroCrossings {
    negative_interval_locations: Vec<f64>,
    negative_intervals: Vec<f64>,
    number_of_negatives: usize,
    positive_interval_locations: Vec<f64>,
    positive_intervals: Vec<f64>,
    number_of_positives: usize,
    peak_interval_locations: Vec<f64>,
    peak_intervals: Vec<f64>,
    number_of_peaks: usize,
    dip_interval_locations: Vec<f64>,
    dip_intervals: Vec<f64>,
    number_of_dips: usize,
}

//---

/// `DesignLowCutFilter()` calculates the coefficients the filter.
fn DesignLowCutFilter(n: usize, fft_size: usize, low_cut_filter: &mut [f64]) {
    for i in 1..=n {
        low_cut_filter[i - 1] = 0.5 - 0.5 * (i as f64 * 2.0 * world::kPi / (n + 1) as f64).cos();
    }
    for i in n..fft_size {
        low_cut_filter[i] = 0.0;
    }
    let mut sum_of_amplitude = 0.0;
    for i in 0..n {
        sum_of_amplitude += low_cut_filter[i];
    }
    for i in 0..n {
        low_cut_filter[i] = -low_cut_filter[i] / sum_of_amplitude;
    }
    for i in 0..(n - 1) / 2 {
        low_cut_filter[fft_size - (n - 1) / 2 + i] = low_cut_filter[i];
    }
    for i in 0..n {
        low_cut_filter[i] = low_cut_filter[i + (n - 1) / 2];
    }
    low_cut_filter[0] += 1.0;
}

/// `GetSpectrumForEstimation()` calculates the spectrum for estimation.
/// This function carries out downsampling to speed up the estimation process
/// and calculates the spectrum of the downsampled signal.
fn GetSpectrumForEstimation(
    x: &[f64],
    x_length: usize,
    y_length: usize,
    actual_fs: f64,
    fft_size: usize,
    decimation_ratio: usize,
    y_spectrum: &mut [fft_complex],
) {
    // Initialization
    let mut y = vec![0.0; fft_size];

    // Downsampling
    if decimation_ratio != 1 {
        decimate(x, x_length, decimation_ratio, &mut y);
    } else {
        for i in 0..x_length {
            y[i] = x[i];
        }
    }

    // Removal of the DC component (y = y - mean value of y)
    let mut mean_y = 0.0;
    for i in 0..y_length {
        mean_y += y[i];
    }
    mean_y /= y_length as f64;
    for i in 0..y_length {
        y[i] -= mean_y;
    }
    for i in y_length..fft_size {
        y[i] = 0.0;
    }

    let mut forward_fft = fft_plan_dft_r2c_1d(fft_size, Some(&y), Some(y_spectrum), FFT_ESTIMATE);
    fft_execute(&mut forward_fft);

    // Low cut filtering (from 0.1.4). Cut off frequency is 50.0 Hz.
    let cutoff_in_sample = matlab_round(actual_fs / world::kCutOff) as usize;
    DesignLowCutFilter(cutoff_in_sample * 2 + 1, fft_size, &mut y);

    let mut filter_spectrum = vec![fft_complex::default(); fft_size];
    let mut forward_fft =
        fft_plan_dft_r2c_1d(fft_size, Some(&y), Some(&mut filter_spectrum), FFT_ESTIMATE);
    fft_execute(&mut forward_fft);

    for i in 0..=fft_size / 2 {
        // Complex number multiplications.
        let tmp =
            y_spectrum[i][0] * filter_spectrum[i][0] - y_spectrum[i][1] * filter_spectrum[i][1];
        y_spectrum[i][1] =
            y_spectrum[i][0] * filter_spectrum[i][1] + y_spectrum[i][1] * filter_spectrum[i][0];
        y_spectrum[i][0] = tmp;
    }
}

/// `GetBestF0Contour()` calculates the best f0 contour based on scores of
/// all candidates. The F0 with highest score is selected.
fn GetBestF0Contour(
    f0_length: usize,
    f0_candidates: &[Vec<f64>],
    f0_scores: &[Vec<f64>],
    number_of_bands: usize,
    best_f0_contour: &mut [f64],
) {
    let mut tmp: f64;
    for i in 0..f0_length {
        tmp = f0_scores[0][i];
        best_f0_contour[i] = f0_candidates[0][i];
        for j in 1..number_of_bands {
            if tmp > f0_scores[j][i] {
                tmp = f0_scores[j][i];
                best_f0_contour[i] = f0_candidates[j][i];
            }
        }
    }
}

/// `FixStep1()` is the 1st step of the postprocessing.
/// This function eliminates the unnatural change of f0 based on allowed_range.
fn FixStep1(
    best_f0_contour: &[f64],
    f0_length: usize,
    voice_range_minimum: usize,
    allowed_range: f64,
    f0_step1: &mut [f64],
) {
    let mut f0_base = vec![0.0; f0_length];
    // Initialization
    for i in voice_range_minimum..f0_length - voice_range_minimum {
        f0_base[i] = best_f0_contour[i];
    }

    // Processing to prevent the jumping of f0
    for i in 0..voice_range_minimum {
        f0_step1[i] = 0.0;
    }
    for i in voice_range_minimum..f0_length {
        f0_step1[i] = if ((f0_base[i] - f0_base[i - 1]) / (world::kMySafeGuardMinimum + f0_base[i]))
            .abs()
            < allowed_range
        {
            f0_base[i]
        } else {
            0.0
        };
    }
}

/// `FixStep2()` is the 2nd step of the postprocessing.
/// This function eliminates the suspected f0 in the anlaut and auslaut.
fn FixStep2(f0_step1: &[f64], f0_length: usize, voice_range_minimum: usize, f0_step2: &mut [f64]) {
    for i in 0..f0_length {
        f0_step2[i] = f0_step1[i];
    }

    let center = (voice_range_minimum - 1) / 2;
    for i in center..f0_length - center {
        for j in 0..=center * 2 {
            if f0_step1[i + j - center] == 0.0 {
                f0_step2[i] = 0.0;
                break;
            }
        }
    }
}

/// `GetNumberOfVoicedSections()` counts the number of voiced sections.
fn GetNumberOfVoicedSections(
    f0: &[f64],
    f0_length: usize,
    positive_index: &mut [usize],
    negative_index: &mut [usize],
    positive_count: &mut usize,
    negative_count: &mut usize,
) {
    *positive_count = 0;
    *negative_count = 0;
    for i in 1..f0_length {
        if f0[i] == 0.0 && f0[i - 1] != 0.0 {
            negative_index[*negative_count] = i - 1;
            *negative_count += 1;
        } else if f0[i - 1] == 0.0 && f0[i] != 0.0 {
            positive_index[*positive_count] = i;
            *positive_count += 1;
        }
    }
}

/// `SelectOneF0()` corrects the `f0[current_index]` based on
/// `f0[current_index + sign]`.
fn SelectBestF0(
    current_f0: f64,
    past_f0: f64,
    f0_candidates: &[Vec<f64>],
    number_of_candidates: usize,
    target_index: usize,
    allowed_range: f64,
) -> f64 {
    let reference_f0 = (current_f0 * 3.0 - past_f0) / 2.0;

    let mut minimum_error = (reference_f0 - f0_candidates[0][target_index]).abs();
    let mut best_f0 = f0_candidates[0][target_index];

    for i in 1..number_of_candidates {
        let current_error = (reference_f0 - f0_candidates[i][target_index]).abs();
        if current_error < minimum_error {
            minimum_error = current_error;
            best_f0 = f0_candidates[i][target_index];
        }
    }
    if (1.0 - best_f0 / reference_f0).abs() > allowed_range {
        return 0.0;
    }
    return best_f0;
}

/// `FixStep3()` is the 3rd step of the postprocessing.
/// This function corrects the f0 candidates from backward to forward.
fn FixStep3(
    f0_step2: &[f64],
    f0_length: usize,
    f0_candidates: &[Vec<f64>],
    number_of_candidates: usize,
    allowed_range: f64,
    negative_index: &[usize],
    negative_count: usize,
    f0_step3: &mut [f64],
) {
    for i in 0..f0_length {
        f0_step3[i] = f0_step2[i];
    }

    for i in 0..negative_count {
        let limit = if i == negative_count - 1 {
            f0_length - 1
        } else {
            negative_index[i + 1]
        };
        for j in negative_index[i]..limit {
            f0_step3[j + 1] = SelectBestF0(
                f0_step3[j],
                f0_step3[j - 1],
                f0_candidates,
                number_of_candidates,
                j + 1,
                allowed_range,
            );
            if f0_step3[j + 1] == 0.0 {
                break;
            }
        }
    }
}

/// `FixStep4()` is the 4th step of the postprocessing.
/// This function corrects the f0 candidates from forward to backward.
fn FixStep4(
    f0_step3: &[f64],
    f0_length: usize,
    f0_candidates: &[Vec<f64>],
    number_of_candidates: usize,
    allowed_range: f64,
    positive_index: &[usize],
    positive_count: usize,
    f0_step4: &mut [f64],
) {
    for i in 0..f0_length {
        f0_step4[i] = f0_step3[i];
    }

    for i in (0..positive_count).rev() {
        let limit = if i == 0 { 1 } else { positive_index[i - 1] };
        for j in (limit + 1..=positive_index[i]).rev() {
            f0_step4[j - 1] = SelectBestF0(
                f0_step4[j],
                f0_step4[j + 1],
                f0_candidates,
                number_of_candidates,
                j - 1,
                allowed_range,
            );
            if f0_step4[j - 1] == 0.0 {
                break;
            }
        }
    }
}

/// `FixF0Contour()` calculates the definitive f0 contour based on all f0
/// candidates. There are four steps.
fn FixF0Contour(
    frame_period: f64,
    number_of_candidates: usize,
    _fs: i32,
    f0_candidates: &[Vec<f64>],
    best_f0_contour: &[f64],
    f0_length: usize,
    f0_floor: f64,
    allowed_range: f64,
    fixed_f0_contour: &mut [f64],
) {
    let voice_range_minimum = (0.5 + 1000.0 / frame_period / f0_floor) as usize * 2 + 1;

    if f0_length <= voice_range_minimum {
        return;
    }

    let mut f0_tmp1 = vec![0.0; f0_length];
    let mut f0_tmp2 = vec![0.0; f0_length];

    FixStep1(
        best_f0_contour,
        f0_length,
        voice_range_minimum,
        allowed_range,
        &mut f0_tmp1,
    );
    FixStep2(&f0_tmp1, f0_length, voice_range_minimum, &mut f0_tmp2);

    let (mut positive_count, mut negative_count) = (0, 0);
    let mut positive_index = vec![0; f0_length];
    let mut negative_index = vec![0; f0_length];
    GetNumberOfVoicedSections(
        &f0_tmp2,
        f0_length,
        &mut positive_index,
        &mut negative_index,
        &mut positive_count,
        &mut negative_count,
    );
    FixStep3(
        &f0_tmp2,
        f0_length,
        f0_candidates,
        number_of_candidates,
        allowed_range,
        &negative_index,
        negative_count,
        &mut f0_tmp1,
    );
    FixStep4(
        &f0_tmp1,
        f0_length,
        f0_candidates,
        number_of_candidates,
        allowed_range,
        &positive_index,
        positive_count,
        fixed_f0_contour,
    );
}

/// `GetFilteredSignal()` calculates the signal that is the convolution of the
/// input signal and low-pass filter.
/// This function is only used in `RawEventByDio()`
fn GetFilteredSignal(
    half_average_length: usize,
    fft_size: usize,
    y_spectrum: &[fft_complex],
    y_length: usize,
    filtered_signal: &mut [f64],
) {
    let mut low_pass_filter = vec![0.0; fft_size];
    // Nuttall window is used as a low-pass filter.
    // Cutoff frequency depends on the window length.
    NuttallWindow(half_average_length * 4, &mut low_pass_filter);

    let mut low_pass_filter_spectrum = vec![fft_complex::default(); fft_size];
    let mut forward_fft = fft_plan_dft_r2c_1d(
        fft_size,
        Some(&low_pass_filter),
        Some(&mut low_pass_filter_spectrum),
        FFT_ESTIMATE,
    );
    fft_execute(&mut forward_fft);

    // Convolution
    let tmp = y_spectrum[0][0] * low_pass_filter_spectrum[0][0]
        - y_spectrum[0][1] * low_pass_filter_spectrum[0][1];
    low_pass_filter_spectrum[0][1] = y_spectrum[0][0] * low_pass_filter_spectrum[0][1]
        + y_spectrum[0][1] * low_pass_filter_spectrum[0][0];
    low_pass_filter_spectrum[0][0] = tmp;
    for i in 1..=fft_size / 2 {
        let tmp = y_spectrum[i][0] * low_pass_filter_spectrum[i][0]
            - y_spectrum[i][1] * low_pass_filter_spectrum[i][1];
        low_pass_filter_spectrum[i][1] = y_spectrum[i][0] * low_pass_filter_spectrum[i][1]
            + y_spectrum[i][1] * low_pass_filter_spectrum[i][0];
        low_pass_filter_spectrum[i][0] = tmp;
        low_pass_filter_spectrum[fft_size - i - 1][0] = low_pass_filter_spectrum[i][0];
        low_pass_filter_spectrum[fft_size - i - 1][1] = low_pass_filter_spectrum[i][1];
    }

    let mut inverse_fft = fft_plan_dft_c2r_1d(
        fft_size,
        Some(&low_pass_filter_spectrum),
        Some(filtered_signal),
        FFT_ESTIMATE,
    );
    fft_execute(&mut inverse_fft);

    // Compensation of the delay.
    let index_bias = half_average_length * 2;
    for i in 0..y_length {
        filtered_signal[i] = filtered_signal[i + index_bias];
    }
}

/// `CheckEvent()` returns `true`, provided that the input value is over `diff`.
#[inline]
fn CheckEvent(x: usize, diff: usize) -> bool {
    x > diff
}

/// `ZeroCrossingEngine()` calculates the zero crossing points from positive to
/// negative. Thanks to Custom.Maid http://custom-made.seesaa.net/ (2012/8/19)
fn ZeroCrossingEngine(
    filtered_signal: &[f64],
    y_length: usize,
    fs: f64,
    interval_locations: &mut [f64],
    intervals: &mut [f64],
) -> usize {
    let mut negative_going_points = vec![0; y_length];

    for i in 0..y_length - 1 {
        negative_going_points[i] = if 0.0 < filtered_signal[i] && filtered_signal[i + 1] <= 0.0 {
            i + 1
        } else {
            0
        };
    }
    negative_going_points[y_length - 1] = 0;

    let mut edges = vec![0; y_length];
    let mut count = 0;
    for i in 0..y_length {
        if negative_going_points[i] > 0 {
            edges[count] = negative_going_points[i];
            count += 1;
        }
    }

    if count < 2 {
        return 0;
    }

    let mut fine_edges = vec![0.0; count];
    for i in 0..count {
        fine_edges[i] = edges[i] as f64
            - filtered_signal[edges[i] - 1]
                / (filtered_signal[edges[i]] - filtered_signal[edges[i] - 1]);
    }

    for i in 0..count - 1 {
        intervals[i] = fs / (fine_edges[i + 1] - fine_edges[i]);
        interval_locations[i] = (fine_edges[i] + fine_edges[i + 1]) / 2.0 / fs;
    }

    return count - 1;
}

/// `GetFourZeroCrossingIntervals()` calculates four zero-crossing intervals.
/// 1. Zero-crossing going from negative to positive.
/// 2. Zero-crossing going from positive to negative.
/// 3. Peak, and
/// 4. dip.
///
/// (3.) and (4.) are calculated from the zero-crossings of
/// the differential of waveform.
fn GetFourZeroCrossingIntervals(
    filtered_signal: &mut [f64],
    y_length: usize,
    actual_fs: f64,
    zero_crossings: &mut ZeroCrossings,
) {
    // x_length / 4 (old version) is fixed at 2013/07/14
    let k_maximum_number: usize = y_length;
    zero_crossings.negative_interval_locations = vec![0.0; k_maximum_number];
    zero_crossings.positive_interval_locations = vec![0.0; k_maximum_number];
    zero_crossings.peak_interval_locations = vec![0.0; k_maximum_number];
    zero_crossings.dip_interval_locations = vec![0.0; k_maximum_number];
    zero_crossings.negative_intervals = vec![0.0; k_maximum_number];
    zero_crossings.positive_intervals = vec![0.0; k_maximum_number];
    zero_crossings.peak_intervals = vec![0.0; k_maximum_number];
    zero_crossings.dip_intervals = vec![0.0; k_maximum_number];

    zero_crossings.number_of_negatives = ZeroCrossingEngine(
        filtered_signal,
        y_length,
        actual_fs,
        &mut zero_crossings.negative_interval_locations,
        &mut zero_crossings.negative_intervals,
    );

    for i in 0..y_length {
        filtered_signal[i] = -filtered_signal[i];
    }
    zero_crossings.number_of_positives = ZeroCrossingEngine(
        filtered_signal,
        y_length,
        actual_fs,
        &mut zero_crossings.positive_interval_locations,
        &mut zero_crossings.positive_intervals,
    );

    for i in 0..y_length - 1 {
        filtered_signal[i] = filtered_signal[i] - filtered_signal[i + 1];
    }
    zero_crossings.number_of_peaks = ZeroCrossingEngine(
        filtered_signal,
        y_length - 1,
        actual_fs,
        &mut zero_crossings.peak_interval_locations,
        &mut zero_crossings.peak_intervals,
    );

    for i in 0..y_length - 1 {
        filtered_signal[i] = -filtered_signal[i];
    }
    zero_crossings.number_of_dips = ZeroCrossingEngine(
        filtered_signal,
        y_length - 1,
        actual_fs,
        &mut zero_crossings.dip_interval_locations,
        &mut zero_crossings.dip_intervals,
    );
}

/// `GetF0CandidateContourSub()` calculates the f0 candidates and deviations.
/// This is the sub-function of `GetF0Candidates()` and assumes the calculation.
fn GetF0CandidateContourSub(
    interpolated_f0_set: &[Vec<f64>],
    f0_length: usize,
    f0_floor: f64,
    f0_ceil: f64,
    boundary_f0: f64,
    f0_candidate: &mut [f64],
    f0_score: &mut [f64],
) {
    for i in 0..f0_length {
        f0_candidate[i] = (interpolated_f0_set[0][i]
            + interpolated_f0_set[1][i]
            + interpolated_f0_set[2][i]
            + interpolated_f0_set[3][i])
            / 4.0;

        f0_score[i] = (((interpolated_f0_set[0][i] - f0_candidate[i])
            * (interpolated_f0_set[0][i] - f0_candidate[i])
            + (interpolated_f0_set[1][i] - f0_candidate[i])
                * (interpolated_f0_set[1][i] - f0_candidate[i])
            + (interpolated_f0_set[2][i] - f0_candidate[i])
                * (interpolated_f0_set[2][i] - f0_candidate[i])
            + (interpolated_f0_set[3][i] - f0_candidate[i])
                * (interpolated_f0_set[3][i] - f0_candidate[i]))
            / 3.0)
            .sqrt();

        if f0_candidate[i] > boundary_f0
            || f0_candidate[i] < boundary_f0 / 2.0
            || f0_candidate[i] > f0_ceil
            || f0_candidate[i] < f0_floor
        {
            f0_candidate[i] = 0.0;
            f0_score[i] = world::kMaximumValue;
        }
    }
}

/// `GetF0CandidateContour()` calculates the F0 candidates based on the
/// zero-crossings.
fn GetF0CandidateContour(
    zero_crossings: &ZeroCrossings,
    boundary_f0: f64,
    f0_floor: f64,
    f0_ceil: f64,
    temporal_positions: &[f64],
    f0_length: usize,
    f0_candidate: &mut [f64],
    f0_score: &mut [f64],
) {
    if !CheckEvent(zero_crossings.number_of_negatives, 2)
        && !CheckEvent(zero_crossings.number_of_positives, 2)
        && !CheckEvent(zero_crossings.number_of_peaks, 2)
        && !CheckEvent(zero_crossings.number_of_dips, 2)
    {
        for i in 0..f0_length {
            f0_score[i] = world::kMaximumValue;
            f0_candidate[i] = 0.0;
        }
        return;
    }

    let mut interpolated_f0_set = [
        vec![0.0; f0_length],
        vec![0.0; f0_length],
        vec![0.0; f0_length],
        vec![0.0; f0_length],
    ];

    interp1(
        &zero_crossings.negative_interval_locations,
        &zero_crossings.negative_intervals,
        zero_crossings.number_of_negatives,
        temporal_positions,
        f0_length,
        &mut interpolated_f0_set[0],
    );
    interp1(
        &zero_crossings.positive_interval_locations,
        &zero_crossings.positive_intervals,
        zero_crossings.number_of_positives,
        temporal_positions,
        f0_length,
        &mut interpolated_f0_set[1],
    );
    interp1(
        &zero_crossings.peak_interval_locations,
        &zero_crossings.peak_intervals,
        zero_crossings.number_of_peaks,
        temporal_positions,
        f0_length,
        &mut interpolated_f0_set[2],
    );
    interp1(
        &zero_crossings.dip_interval_locations,
        &zero_crossings.dip_intervals,
        zero_crossings.number_of_dips,
        temporal_positions,
        f0_length,
        &mut interpolated_f0_set[3],
    );

    GetF0CandidateContourSub(
        &interpolated_f0_set,
        f0_length,
        f0_floor,
        f0_ceil,
        boundary_f0,
        f0_candidate,
        f0_score,
    );
}

/// `GetF0CandidateFromRawEvent()` calculates F0 candidate contour in 1-ch signal
fn GetF0CandidateFromRawEvent(
    boundary_f0: f64,
    fs: f64,
    y_spectrum: &[fft_complex],
    y_length: usize,
    fft_size: usize,
    f0_floor: f64,
    f0_ceil: f64,
    temporal_positions: &[f64],
    f0_length: usize,
    f0_score: &mut [f64],
    f0_candidate: &mut [f64],
) {
    let mut filtered_signal = vec![0.0; fft_size];
    GetFilteredSignal(
        matlab_round(fs / boundary_f0 / 2.0) as usize,
        fft_size,
        y_spectrum,
        y_length,
        &mut filtered_signal,
    );

    let mut zero_crossings = ZeroCrossings::default();
    GetFourZeroCrossingIntervals(&mut filtered_signal, y_length, fs, &mut zero_crossings);

    GetF0CandidateContour(
        &zero_crossings,
        boundary_f0,
        f0_floor,
        f0_ceil,
        temporal_positions,
        f0_length,
        f0_candidate,
        f0_score,
    );
}

/// `GetF0CandidatesAndScores()` calculates all f0 candidates and their scores.
fn GetF0CandidatesAndScores(
    boundary_f0_list: &[f64],
    number_of_bands: usize,
    actual_fs: f64,
    y_length: usize,
    temporal_positions: &[f64],
    f0_length: usize,
    y_spectrum: &[fft_complex],
    fft_size: usize,
    f0_floor: f64,
    f0_ceil: f64,
    raw_f0_candidates: &mut [Vec<f64>],
    raw_f0_scores: &mut [Vec<f64>],
) {
    let mut f0_candidate = vec![0.0; f0_length];
    let mut f0_score = vec![0.0; f0_length];

    // Calculation of the acoustics events (zero-crossing)
    for i in 0..number_of_bands {
        GetF0CandidateFromRawEvent(
            boundary_f0_list[i],
            actual_fs,
            y_spectrum,
            y_length,
            fft_size,
            f0_floor,
            f0_ceil,
            temporal_positions,
            f0_length,
            &mut f0_score,
            &mut f0_candidate,
        );
        for j in 0..f0_length {
            // A way to avoid zero division
            raw_f0_scores[i][j] = f0_score[j] / (f0_candidate[j] + world::kMySafeGuardMinimum);
            raw_f0_candidates[i][j] = f0_candidate[j];
        }
    }
}

/// `DioGeneralBody()` estimates the F0 based on Distributed Inline-filter
/// Operation.
fn DioGeneralBody(
    x: &[f64],
    x_length: usize,
    fs: i32,
    frame_period: f64,
    f0_floor: f64,
    f0_ceil: f64,
    channels_in_octave: f64,
    speed: u8,
    allowed_range: f64,
    temporal_positions: &mut [f64],
    f0: &mut [f64],
) {
    let number_of_bands =
        1 + ((f0_ceil / f0_floor).ln() / world::kLog2 * channels_in_octave) as usize;
    let mut boundary_f0_list = vec![0.0; number_of_bands];
    for i in 0..number_of_bands {
        boundary_f0_list[i] = f0_floor * 2.0_f64.powf((i + 1) as f64 / channels_in_octave);
    }

    // normalization
    let decimation_ratio = MyMax(MyMin(speed, 12), 1);
    let y_length = 1 + (x_length / decimation_ratio as usize);
    let actual_fs = fs as f64 / decimation_ratio as f64;
    let fft_size = GetSuitableFFTSize(
        y_length
            + matlab_round(actual_fs / world::kCutOff) as usize * 2
            + 1
            + 4 * (1.0 + actual_fs / boundary_f0_list[0] / 2.0) as usize,
    );

    // Calculation of the spectrum used for the f0 estimation
    let mut y_spectrum = vec![fft_complex::default(); fft_size];
    GetSpectrumForEstimation(
        x,
        x_length,
        y_length,
        actual_fs,
        fft_size,
        decimation_ratio as usize,
        &mut y_spectrum,
    );

    let f0_length = GetSamplesForDIO(fs, x_length, frame_period);
    let mut f0_candidates = vec![vec![0.0; f0_length]; number_of_bands];
    let mut f0_scores = vec![vec![0.0; f0_length]; number_of_bands];

    for i in 0..f0_length {
        temporal_positions[i] = i as f64 * frame_period / 1000.0;
    }

    GetF0CandidatesAndScores(
        &boundary_f0_list,
        number_of_bands,
        actual_fs,
        y_length,
        temporal_positions,
        f0_length,
        &y_spectrum,
        fft_size,
        f0_floor,
        f0_ceil,
        &mut f0_candidates,
        &mut f0_scores,
    );

    // Selection of the best value based on fundamental-ness.
    // This function is related with SortCandidates() in MATLAB.
    let mut best_f0_contour = vec![0.0; f0_length];
    GetBestF0Contour(
        f0_length,
        &f0_candidates,
        &f0_scores,
        number_of_bands,
        &mut best_f0_contour,
    );

    // Postprocessing to find the best f0-contour.
    FixF0Contour(
        frame_period,
        number_of_bands,
        fs,
        &f0_candidates,
        &best_f0_contour,
        f0_length,
        f0_floor,
        allowed_range,
        f0,
    );
}

//---

/// Struct for DIO
pub struct DioOption {
    f0_floor: f64,
    f0_ceil: f64,
    channels_in_octave: f64,
    frame_period: f64,  // msec
    speed: u8,          // (1, 2, ..., 12)
    allowed_range: f64, // Threshold used for fixing the F0 contour.
}

/// DIO
///
/// - Input
///     - `x`                   : Input signal
///     - `x_length`            : Length of `x`
///     - `fs`                  : Sampling frequency
///     - `option`              : Struct to order the parameter for DIO
///
/// - Output
///     - `temporal_positions`  : Temporal positions.
///     - `f0`                  : F0 contour.
pub fn Dio(
    x: &[f64],
    x_length: usize,
    fs: i32,
    option: &DioOption,
    temporal_positions: &mut [f64],
    f0: &mut [f64],
) {
    DioGeneralBody(
        x,
        x_length,
        fs,
        option.frame_period,
        option.f0_floor,
        option.f0_ceil,
        option.channels_in_octave,
        option.speed,
        option.allowed_range,
        temporal_positions,
        f0,
    );
}

impl DioOption {
    /// Sets the default parameters.
    ///
    /// - Output
    ///     - Struct for the optional parameter.
    ///
    /// You can change default parameters.
    pub fn new() -> Self {
        Self {
            channels_in_octave: 2.0,
            f0_ceil: world::kCeilF0,
            f0_floor: world::kFloorF0,
            frame_period: 5.0,

            speed: 1,

            allowed_range: 0.1,
        }
    }

    pub fn channels_in_octave(self, channels_in_octave: f64) -> Self {
        Self {
            channels_in_octave,
            ..self
        }
    }

    pub fn f0_ceil(self, f0_ceil: f64) -> Self {
        Self { f0_ceil, ..self }
    }

    pub fn f0_floor(self, f0_floor: f64) -> Self {
        Self { f0_floor, ..self }
    }

    pub fn frame_period(self, frame_period: f64) -> Self {
        Self {
            frame_period,
            ..self
        }
    }

    /// You can use the value from `1` to `12`.
    /// Default value `11` is for the `fs` of 44.1 kHz.
    /// The lower value you use, the better performance you can obtain.
    pub fn speed(self, speed: u8) -> Self {
        Self { speed, ..self }
    }

    /// You can give a positive real number as the threshold.
    /// The most strict value is `0`, and there is no upper limit.
    /// On the other hand, I think that the value from `0.02` to `0.2` is reasonable.
    pub fn allowed_range(self, allowed_range: f64) -> Self {
        Self {
            allowed_range,
            ..self
        }
    }
}

/// `GetSamplesForDIO()` calculates the number of samples required for `Dio()`.
///
/// - Input
///     - `fs`              : Sampling frequency [Hz]
///     - `x_length`        : Length of the input signal [Sample].
///     - `frame_period`    : Frame shift [msec]
///
/// - Output
///     - The number of samples required to store the results of `Dio()`
pub fn GetSamplesForDIO(fs: i32, x_length: usize, frame_period: f64) -> usize {
    (1000.0 * x_length as f64 / fs as f64 / frame_period) as usize + 1
}
