//! F0 estimation based on Harvest.

use crate::common::{FFTProsess, ForwardRealFFT, GetSuitableFFTSize, MyMax, MyMin, NuttallWindow};
use crate::constantnumbers::world;
use crate::fft::{
    fft_complex, fft_execute, fft_plan_dft_c2r_1d, fft_plan_dft_r2c_1d, FFT_ESTIMATE,
};
use crate::matlabfunctions::{decimate, interp1, matlab_round};

/// struct for `RawEventByHarvest()`
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

// Since the waveform of beginning and ending after decimate include noise,
// the input waveform is extended. This is the processing for the
// compatibility with MATLAB version.
fn GetWaveformAndSpectrumSub(
    x: &[f64],
    x_length: usize,
    y_length: usize,
    _actual_fs: f64,
    decimation_ratio: usize,
    y: &mut [f64],
) {
    if decimation_ratio == 1 {
        for i in 0..x_length {
            y[i] = x[i];
        }
        return;
    }

    let lag = ((140.0 / decimation_ratio as f64).ceil() * decimation_ratio as f64) as usize;
    let new_x_length = x_length + lag * 2;
    let mut new_y: Vec<f64> = vec![0.0; new_x_length];
    for i in 0..new_x_length {
        new_y[i] = 0.0;
    }
    let mut new_x: Vec<f64> = vec![0.0; new_x_length];
    for i in 0..lag {
        new_x[i] = x[0];
    }
    for i in lag..lag + x_length {
        new_x[i] = x[i - lag];
    }
    for i in lag + x_length..new_x_length {
        new_x[i] = x[x_length - 1];
    }

    decimate(&new_x, new_x_length, decimation_ratio, &mut new_y);
    for i in 0..y_length {
        y[i] = new_y[lag / decimation_ratio + i];
    }
}

/// `GetWaveformAndSpectrum()` calculates the downsampled signal and its spectrum
fn GetWaveformAndSpectrum(
    x: &[f64],
    x_length: usize,
    y_length: usize,
    actual_fs: f64,
    fft_size: usize,
    decimation_ratio: usize,
    y: &mut [f64],
    y_spectrum: &mut [fft_complex],
) {
    // Initialization
    for i in 0..fft_size {
        y[i] = 0.0;
    }

    // Processing for the compatibility with MATLAB version
    GetWaveformAndSpectrumSub(x, x_length, y_length, actual_fs, decimation_ratio, y);

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

    let mut forward_fft = fft_plan_dft_r2c_1d(fft_size, Some(y), Some(y_spectrum), FFT_ESTIMATE);
    fft_execute(&mut forward_fft);
}

/// `GetFilteredSignal()` calculates the signal that is the convolution of the
/// input signal and band-pass filter.
fn GetFilteredSignal(
    boundary_f0: f64,
    fft_size: usize,
    fs: f64,
    y_spectrum: &[fft_complex],
    y_length: usize,
    filtered_signal: &mut [f64],
) {
    let filter_length_half = matlab_round(fs / boundary_f0 * 2.0);
    let mut band_pass_filter: Vec<f64> = vec![0.0; fft_size];
    NuttallWindow(filter_length_half as usize * 2 + 1, &mut band_pass_filter);
    for i in -filter_length_half..=filter_length_half {
        band_pass_filter[(i + filter_length_half) as usize] *=
            (2.0 * world::kPi * boundary_f0 * i as f64 / fs).cos();
    }
    for i in filter_length_half as usize * 2 + 1..fft_size {
        band_pass_filter[i] = 0.0;
    }

    let mut band_pass_filter_spectrum: Vec<fft_complex> = vec![fft_complex::default(); fft_size];
    let mut forward_fft = fft_plan_dft_r2c_1d(
        fft_size,
        Some(&band_pass_filter),
        Some(&mut band_pass_filter_spectrum),
        FFT_ESTIMATE,
    );
    fft_execute(&mut forward_fft);

    // Convolution
    let mut tmp = y_spectrum[0][0] * band_pass_filter_spectrum[0][0]
        - y_spectrum[0][1] * band_pass_filter_spectrum[0][1];
    band_pass_filter_spectrum[0][1] = y_spectrum[0][0] * band_pass_filter_spectrum[0][1]
        + y_spectrum[0][1] * band_pass_filter_spectrum[0][0];
    band_pass_filter_spectrum[0][0] = tmp;
    for i in 1..=fft_size / 2 {
        tmp = y_spectrum[i][0] * band_pass_filter_spectrum[i][0]
            - y_spectrum[i][1] * band_pass_filter_spectrum[i][1];
        band_pass_filter_spectrum[i][1] = y_spectrum[i][0] * band_pass_filter_spectrum[i][1]
            + y_spectrum[i][1] * band_pass_filter_spectrum[i][0];
        band_pass_filter_spectrum[i][0] = tmp;
        band_pass_filter_spectrum[fft_size - i - 1][0] = band_pass_filter_spectrum[i][0];
        band_pass_filter_spectrum[fft_size - i - 1][1] = band_pass_filter_spectrum[i][1];
    }

    let mut inverse_fft = fft_plan_dft_c2r_1d(
        fft_size,
        Some(&band_pass_filter_spectrum),
        Some(filtered_signal),
        FFT_ESTIMATE,
    );
    fft_execute(&mut inverse_fft);

    // Compensation of the delay.
    let index_bias = filter_length_half as usize + 1;
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
/// negative.
fn ZeroCrossingEngine(
    filtered_signal: &[f64],
    y_length: usize,
    fs: f64,
    interval_locations: &mut [f64],
    intervals: &mut [f64],
) -> usize {
    let mut negative_going_points: Vec<usize> = vec![0; y_length];

    for i in 0..y_length - 1 {
        negative_going_points[i] = if 0.0 < filtered_signal[i] && filtered_signal[i + 1] <= 0.0 {
            i + 1
        } else {
            0
        };
    }
    negative_going_points[y_length - 1] = 0;

    let mut edges: Vec<usize> = vec![0; y_length];
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

    let mut fine_edges: Vec<f64> = vec![0.0; count];
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
/// (1) Zero-crossing going from negative to positive.
/// (2) Zero-crossing going from positive to negative.
/// (3) Peak, and (4) dip. (3) and (4) are calculated from the zero-crossings of
/// the differential of waveform.
fn GetFourZeroCrossingIntervals(
    filtered_signal: &mut [f64],
    y_length: usize,
    actual_fs: f64,
    zero_crossings: &mut ZeroCrossings,
) {
    let maximum_number = y_length;
    zero_crossings.negative_interval_locations = vec![0.0; maximum_number];
    zero_crossings.positive_interval_locations = vec![0.0; maximum_number];
    zero_crossings.peak_interval_locations = vec![0.0; maximum_number];
    zero_crossings.dip_interval_locations = vec![0.0; maximum_number];
    zero_crossings.negative_intervals = vec![0.0; maximum_number];
    zero_crossings.positive_intervals = vec![0.0; maximum_number];
    zero_crossings.peak_intervals = vec![0.0; maximum_number];
    zero_crossings.dip_intervals = vec![0.0; maximum_number];

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

fn GetF0CandidateContourSub(
    interpolated_f0_set: &[Vec<f64>],
    f0_length: usize,
    f0_floor: f64,
    f0_ceil: f64,
    boundary_f0: f64,
    f0_candidate: &mut [f64],
) {
    let upper = boundary_f0 * 1.1;
    let lower = boundary_f0 * 0.9;
    for i in 0..f0_length {
        f0_candidate[i] = (interpolated_f0_set[0][i]
            + interpolated_f0_set[1][i]
            + interpolated_f0_set[2][i]
            + interpolated_f0_set[3][i])
            / 4.0;

        if f0_candidate[i] > upper
            || f0_candidate[i] < lower
            || f0_candidate[i] > f0_ceil
            || f0_candidate[i] < f0_floor
        {
            f0_candidate[i] = 0.0;
        }
    }
}

/// `GetF0CandidateContour()` calculates the F0 candidate contour in 1-ch signal.
/// Calculation of F0 candidates is carried out in `GetF0CandidatesSub()`.
fn GetF0CandidateContour(
    zero_crossings: &ZeroCrossings,
    boundary_f0: f64,
    f0_floor: f64,
    f0_ceil: f64,
    temporal_positions: &[f64],
    f0_length: usize,
    f0_candidate: &mut [f64],
) {
    if !CheckEvent(zero_crossings.number_of_negatives, 2)
        && !CheckEvent(zero_crossings.number_of_positives, 2)
        && !CheckEvent(zero_crossings.number_of_peaks, 2)
        && !CheckEvent(zero_crossings.number_of_dips, 2)
    {
        for i in 0..f0_length {
            f0_candidate[i] = 0.0;
        }
        return;
    }

    let mut interpolated_f0_set: [Vec<f64>; 4] = [
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
    );
}

// GetF0CandidateFromRawEvent() f0 candidate contour in 1-ch signal
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
    f0_candidate: &mut [f64],
) {
    let mut filtered_signal: Vec<f64> = vec![0.0; fft_size];
    GetFilteredSignal(
        boundary_f0,
        fft_size,
        fs,
        y_spectrum,
        y_length,
        &mut filtered_signal,
    );

    let mut zero_crossings: ZeroCrossings = Default::default();
    GetFourZeroCrossingIntervals(&mut filtered_signal, y_length, fs, &mut zero_crossings);

    GetF0CandidateContour(
        &zero_crossings,
        boundary_f0,
        f0_floor,
        f0_ceil,
        &temporal_positions,
        f0_length,
        f0_candidate,
    );
}

/// `GetRawF0Candidates()` calculates f0 candidates in all channels.
fn GetRawF0Candidates(
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
) {
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
            &mut raw_f0_candidates[i],
        );
    }
}

/// `DetectF0CandidatesSub1()` calculates VUV areas.
fn DetectOfficialF0CandidatesSub1(
    vuv: &[i32],
    number_of_channels: usize,
    st: &mut [usize],
    ed: &mut [usize],
) -> usize {
    let mut number_of_voiced_sections = 0;
    let mut tmp: i32;
    for i in 1..number_of_channels {
        tmp = vuv[i] - vuv[i - 1];
        if tmp == 1 {
            st[number_of_voiced_sections] = i;
        }
        if tmp == -1 {
            ed[number_of_voiced_sections] = i;
            number_of_voiced_sections += 1;
        }
    }

    return number_of_voiced_sections;
}

/// `DetectOfficialF0CandidatesSub2()` calculates F0 candidates in a frame
fn DetectOfficialF0CandidatesSub2(
    _vuv: &[i32],
    raw_f0_candidates: &[Vec<f64>],
    index: usize,
    number_of_voiced_sections: usize,
    st: &[usize],
    ed: &[usize],
    max_candidates: usize,
    f0_list: &mut [f64],
) -> usize {
    let mut number_of_candidates = 0;
    let mut tmp_f0: f64;
    for i in 0..number_of_voiced_sections {
        if ed[i] - st[i] < 10 {
            continue;
        }

        tmp_f0 = 0.0;
        for j in st[i]..ed[i] {
            tmp_f0 += raw_f0_candidates[j][index];
        }
        tmp_f0 /= (ed[i] - st[i]) as f64;
        f0_list[number_of_candidates] = tmp_f0;
        number_of_candidates += 1;
    }

    for i in number_of_candidates..max_candidates {
        f0_list[i] = 0.0;
    }
    return number_of_candidates;
}

/// `DetectOfficialF0Candidates()` detectes F0 candidates from multi-channel
/// candidates.
fn DetectOfficialF0Candidates(
    raw_f0_candidates: &[Vec<f64>],
    number_of_channels: usize,
    f0_length: usize,
    max_candidates: usize,
    f0_candidates: &mut [Vec<f64>],
) -> usize {
    let mut number_of_candidates = 0;

    let mut vuv: Vec<i32> = vec![0; number_of_channels];
    let mut st: Vec<usize> = vec![0; number_of_channels];
    let mut ed: Vec<usize> = vec![0; number_of_channels];
    let mut number_of_voiced_sections;
    for i in 0..f0_length {
        for j in 0..number_of_channels {
            vuv[j] = if raw_f0_candidates[j][i] > 0.0 { 1 } else { 0 };
        }
        vuv[0] = 0;
        vuv[number_of_channels - 1] = 0;
        number_of_voiced_sections =
            DetectOfficialF0CandidatesSub1(&vuv, number_of_channels, &mut st, &mut ed);
        number_of_candidates = MyMax(
            number_of_candidates,
            DetectOfficialF0CandidatesSub2(
                &vuv,
                raw_f0_candidates,
                i,
                number_of_voiced_sections,
                &st,
                &ed,
                max_candidates,
                &mut f0_candidates[i],
            ),
        );
    }

    return number_of_candidates;
}

/// `OverlapF0Candidates()` spreads the candidates to anteroposterior frames.
fn OverlapF0Candidates(
    f0_length: usize,
    number_of_candidates: usize,
    f0_candidates: &mut [Vec<f64>],
) {
    let n = 3;
    for i in 1..=n {
        for j in 0..number_of_candidates {
            for k in i..f0_length {
                f0_candidates[k][j + (number_of_candidates * i)] = f0_candidates[k - i][j];
            }
            for k in 0..f0_length - i {
                f0_candidates[k][j + (number_of_candidates * (i + n))] = f0_candidates[k + i][j];
            }
        }
    }
}

/// `GetBaseIndex()` calculates the temporal positions for windowing.
fn GetBaseIndex(
    current_position: f64,
    base_time: &[f64],
    base_time_length: usize,
    fs: f64,
    base_index: &mut [isize],
) {
    // First-aid treatment
    let basic_index = matlab_round((current_position + base_time[0]) * fs + 0.001) as isize;

    for i in 0..base_time_length {
        base_index[i] = basic_index + i as isize;
    }
}

/// `GetMainWindow()` generates the window function.
fn GetMainWindow(
    current_position: f64,
    base_index: &[isize],
    base_time_length: usize,
    fs: f64,
    window_length_in_time: f64,
    main_window: &mut [f64],
) {
    let mut tmp;
    for i in 0..base_time_length {
        tmp = (base_index[i] as f64 - 1.0) / fs - current_position;
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
    base_index: &[isize],
    main_window: &[f64],
    diff_window: &[f64],
    base_time_length: usize,
    forward_real_fft: &mut ForwardRealFFT,
    main_spectrum: &mut [fft_complex],
    diff_spectrum: &mut [fft_complex],
) {
    let mut safe_index: Vec<usize> = vec![0; base_time_length];

    for i in 0..base_time_length {
        safe_index[i] = MyMax(0, MyMin(x_length as isize - 1, base_index[i] - 1)) as usize;
    }
    for i in 0..base_time_length {
        forward_real_fft.waveform[i] = x[safe_index[i]] * main_window[i];
    }
    for i in base_time_length..fft_size {
        forward_real_fft.waveform[i] = 0.0;
    }

    // fft_execute(&mut forward_real_fft.forward_fft);
    forward_real_fft.exec();
    for i in 0..=fft_size / 2 {
        main_spectrum[i][0] = forward_real_fft.spectrum[i][0];
        main_spectrum[i][1] = forward_real_fft.spectrum[i][1];
    }

    for i in 0..base_time_length {
        forward_real_fft.waveform[i] = x[safe_index[i]] * diff_window[i];
    }
    for i in base_time_length..fft_size {
        forward_real_fft.waveform[i] = 0.0;
    }
    // fft_execute(&mut forward_real_fft.forward_fft);
    forward_real_fft.exec();
    for i in 0..=fft_size / 2 {
        diff_spectrum[i][0] = forward_real_fft.spectrum[i][0];
        diff_spectrum[i][1] = forward_real_fft.spectrum[i][1];
    }
}

fn FixF0(
    power_spectrum: &[f64],
    numerator_i: &[f64],
    fft_size: usize,
    fs: f64,
    current_f0: f64,
    number_of_harmonics: usize,
    refined_f0: &mut f64,
    score: &mut f64,
) {
    let mut amplitude_list: Vec<f64> = vec![0.0; number_of_harmonics];
    let mut instantaneous_frequency_list: Vec<f64> = vec![0.0; number_of_harmonics];

    let mut index;
    for i in 0..number_of_harmonics {
        index = matlab_round(current_f0 * fft_size as f64 / fs * (i as f64 + 1.0)) as usize;
        instantaneous_frequency_list[i] = if power_spectrum[index] == 0.0 {
            0.0
        } else {
            index as f64 * fs / fft_size as f64
                + numerator_i[index] / power_spectrum[index] * fs / 2.0 / world::kPi
        };
        amplitude_list[i] = (power_spectrum[index]).sqrt();
    }
    let mut denominator = 0.0;
    let mut numerator = 0.0;
    *score = 0.0;
    for i in 0..number_of_harmonics {
        numerator += amplitude_list[i] * instantaneous_frequency_list[i];
        denominator += amplitude_list[i] * (i as f64 + 1.0);
        *score +=
            ((instantaneous_frequency_list[i] / (i as f64 + 1.0) - current_f0) / current_f0).abs();
    }

    *refined_f0 = numerator / (denominator + world::kMySafeGuardMinimum);
    *score = 1.0 / (*score / number_of_harmonics as f64 + world::kMySafeGuardMinimum);
}

/// `GetMeanF0()` calculates the instantaneous frequency.
fn GetMeanF0(
    x: &[f64],
    x_length: usize,
    fs: f64,
    current_position: f64,
    current_f0: f64,
    fft_size: usize,
    window_length_in_time: f64,
    base_time: &[f64],
    base_time_length: usize,
    refined_f0: &mut f64,
    refined_score: &mut f64,
) {
    let mut forward_real_fft = ForwardRealFFT::new(fft_size);
    let mut main_spectrum: Vec<fft_complex> = vec![fft_complex::default(); fft_size]; //[0.0,0.0]
    let mut diff_spectrum: Vec<fft_complex> = vec![fft_complex::default(); fft_size];

    let mut base_index: Vec<isize> = vec![0; base_time_length];
    let mut main_window: Vec<f64> = vec![0.0; base_time_length];
    let mut diff_window: Vec<f64> = vec![0.0; base_time_length];

    GetBaseIndex(
        current_position,
        base_time,
        base_time_length,
        fs,
        &mut base_index,
    );
    GetMainWindow(
        current_position,
        &base_index,
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
        &base_index,
        &main_window,
        &diff_window,
        base_time_length,
        &mut forward_real_fft,
        &mut main_spectrum,
        &mut diff_spectrum,
    );

    let mut power_spectrum: Vec<f64> = vec![0.0; fft_size / 2 + 1];
    let mut numerator_i: Vec<f64> = vec![0.0; fft_size / 2 + 1];
    for j in 0..=fft_size / 2 {
        numerator_i[j] =
            main_spectrum[j][0] * diff_spectrum[j][1] - main_spectrum[j][1] * diff_spectrum[j][0];
        power_spectrum[j] =
            main_spectrum[j][0] * main_spectrum[j][0] + main_spectrum[j][1] * main_spectrum[j][1];
    }

    let number_of_harmonics = MyMin((fs / 2.0 / current_f0) as usize, 6);
    FixF0(
        &power_spectrum,
        &numerator_i,
        fft_size,
        fs,
        current_f0,
        number_of_harmonics,
        refined_f0,
        refined_score,
    );
}

/// `GetRefinedF0()` calculates F0 and its score based on instantaneous frequency.
fn GetRefinedF0(
    x: &[f64],
    x_length: usize,
    fs: f64,
    current_position: f64,
    current_f0: f64,
    f0_floor: f64,
    f0_ceil: f64,
    refined_f0: &mut f64,
    refined_score: &mut f64,
) {
    if current_f0 <= 0.0 {
        *refined_f0 = 0.0;
        *refined_score = 0.0;
        return;
    }

    let half_window_length: usize = (1.5 * fs / current_f0 + 1.0) as usize;
    let window_length_in_time = (2.0 * half_window_length as f64 + 1.0) / fs;
    let mut base_time: Vec<f64> = vec![0.0; half_window_length * 2 + 1];
    for i in 0..half_window_length * 2 + 1 {
        base_time[i] = (-(half_window_length as isize) + i as isize) as f64 / fs;
    }
    let fft_size = (2.0_f64
        .powf(2.0 + (((half_window_length as f64 * 2.0 + 1.0).ln() / world::kLog2) as i32) as f64))
        as usize;

    GetMeanF0(
        x,
        x_length,
        fs,
        current_position,
        current_f0,
        fft_size,
        window_length_in_time,
        &base_time,
        half_window_length * 2 + 1,
        refined_f0,
        refined_score,
    );

    if *refined_f0 < f0_floor || *refined_f0 > f0_ceil || *refined_score < 2.5 {
        *refined_f0 = 0.0;
        *refined_score = 0.0;
    }
}

/// `RefineF0()` modifies the F0 by instantaneous frequency.
fn RefineF0Candidates(
    x: &[f64],
    x_length: usize,
    fs: f64,
    temporal_positions: &[f64],
    f0_length: usize,
    max_candidates: usize,
    f0_floor: f64,
    f0_ceil: f64,
    refined_f0_candidates: &mut [Vec<f64>],
    f0_scores: &mut [Vec<f64>],
) {
    for i in 0..f0_length {
        for j in 0..max_candidates {
            GetRefinedF0(
                x,
                x_length,
                fs,
                temporal_positions[i],
                refined_f0_candidates[i][j],
                f0_floor,
                f0_ceil,
                &mut refined_f0_candidates[i][j],
                &mut f0_scores[i][j],
            );
        }
    }
}

/// `SelectBestF0()` obtains the nearlest F0 in reference_f0.
fn SelectBestF0(
    reference_f0: f64,
    f0_candidates: &[f64],
    number_of_candidates: usize,
    allowed_range: f64,
    best_error: &mut f64,
) -> f64 {
    let mut best_f0 = 0.0;
    *best_error = allowed_range;

    let mut tmp: f64;
    for i in 0..number_of_candidates {
        tmp = (reference_f0 - f0_candidates[i]).abs() / reference_f0;
        if tmp > *best_error {
            continue;
        }
        best_f0 = f0_candidates[i];
        *best_error = tmp;
    }

    return best_f0;
}

fn RemoveUnreliableCandidatesSub(
    i: usize,
    j: usize,
    tmp_f0_candidates: &[Vec<f64>],
    number_of_candidates: usize,
    f0_candidates: &mut [Vec<f64>],
    f0_scores: &mut [Vec<f64>],
) {
    let reference_f0 = f0_candidates[i][j];
    let mut error1 = 0.0;
    let mut error2 = 0.0;
    let threshold = 0.05;
    if reference_f0 == 0.0 {
        return;
    }
    SelectBestF0(
        reference_f0,
        &tmp_f0_candidates[i + 1],
        number_of_candidates,
        1.0,
        &mut error1,
    );
    SelectBestF0(
        reference_f0,
        &tmp_f0_candidates[i - 1],
        number_of_candidates,
        1.0,
        &mut error2,
    );
    let min_error = MyMin(error1, error2);
    if min_error <= threshold {
        return;
    };
    f0_candidates[i][j] = 0.0;
    f0_scores[i][j] = 0.0;
}

/// `RemoveUnreliableCandidates()`.
fn RemoveUnreliableCandidates(
    f0_length: usize,
    number_of_candidates: usize,
    f0_candidates: &mut [Vec<f64>],
    f0_scores: &mut [Vec<f64>],
) {
    let mut tmp_f0_candidates: Vec<Vec<f64>> = vec![vec![0.0; number_of_candidates]; f0_length];
    for i in 0..f0_length {
        for j in 0..number_of_candidates {
            tmp_f0_candidates[i][j] = f0_candidates[i][j];
        }
    }

    for i in 1..f0_length - 1 {
        for j in 0..number_of_candidates {
            RemoveUnreliableCandidatesSub(
                i,
                j,
                &tmp_f0_candidates,
                number_of_candidates,
                f0_candidates,
                f0_scores,
            );
        }
    }
}

/// `SearchF0Base()` gets the F0 with the highest score.
fn SearchF0Base(
    f0_candidates: &[Vec<f64>],
    f0_scores: &[Vec<f64>],
    f0_length: usize,
    number_of_candidates: usize,
    base_f0_contour: &mut [f64],
) {
    let mut tmp_best_score: f64;
    for i in 0..f0_length {
        base_f0_contour[i] = 0.0;
        tmp_best_score = 0.0;
        for j in 0..number_of_candidates {
            if f0_scores[i][j] > tmp_best_score {
                base_f0_contour[i] = f0_candidates[i][j];
                tmp_best_score = f0_scores[i][j];
            }
        }
    }
}

/// Step 1: Rapid change of F0 contour is replaced by 0.
fn FixStep1(f0_base: &[f64], f0_length: usize, allowed_range: f64, f0_step1: &mut [f64]) {
    for i in 0..f0_length {
        f0_step1[i] = 0.0;
    }
    let mut reference_f0: f64;
    for i in 2..f0_length {
        if f0_base[i] == 0.0 {
            continue;
        }
        reference_f0 = f0_base[i - 1] * 2.0 - f0_base[i - 2];
        f0_step1[i] = if ((f0_base[i] - reference_f0) / reference_f0).abs() > allowed_range
            && (f0_base[i] - f0_base[i - 1]).abs() / f0_base[i - 1] > allowed_range
        {
            0.0
        } else {
            f0_base[i]
        };
    }
}

/// `GetBoundaryList()` detects boundaries between voiced and unvoiced sections.
fn GetBoundaryList(f0: &[f64], f0_length: usize, boundary_list: &mut [usize]) -> usize {
    let mut number_of_boundaries = 0;
    let mut vuv: Vec<i32> = vec![0; f0_length];
    for i in 0..f0_length {
        vuv[i] = if f0[i] > 0.0 { 1 } else { 0 };
    }
    vuv[0] = 0;
    vuv[f0_length - 1] = 0;

    for i in 1..f0_length {
        if vuv[i] - vuv[i - 1] != 0 {
            boundary_list[number_of_boundaries] = i - number_of_boundaries % 2;
            number_of_boundaries += 1;
        }
    }

    return number_of_boundaries;
}

/// Step 2: Voiced sections with a short period are removed.
fn FixStep2(f0_step1: &[f64], f0_length: usize, voice_range_minimum: usize, f0_step2: &mut [f64]) {
    for i in 0..f0_length {
        f0_step2[i] = f0_step1[i];
    }
    let mut boundary_list: Vec<usize> = vec![0; f0_length];
    let number_of_boundaries = GetBoundaryList(f0_step1, f0_length, &mut boundary_list);

    for i in 0..number_of_boundaries / 2 {
        if boundary_list[i * 2 + 1] - boundary_list[i * 2] >= voice_range_minimum {
            continue;
        }
        for j in boundary_list[i * 2]..=boundary_list[(i * 2) + 1] {
            f0_step2[j] = 0.0;
        }
    }
}

/// `GetMultiChannelF0()` separates each voiced section into independent channel.
fn GetMultiChannelF0(
    f0: &[f64],
    f0_length: usize,
    boundary_list: &[usize],
    number_of_boundaries: usize,
    multi_channel_f0: &mut [Vec<f64>],
) {
    for i in 0..number_of_boundaries / 2 {
        for j in 0..boundary_list[i * 2] {
            multi_channel_f0[i][j] = 0.0;
        }
        for j in boundary_list[i * 2]..=boundary_list[i * 2 + 1] {
            multi_channel_f0[i][j] = f0[j];
        }
        for j in boundary_list[i * 2 + 1] + 1..f0_length {
            multi_channel_f0[i][j] = 0.0;
        }
    }
}

/// (In the context of C++)
/// `abs()` often causes bugs, an original function is used.
#[inline]
#[allow(dead_code)]
fn MyAbsInt(x: i32) -> i32 {
    return if x > 0 { x } else { -x };
}

/// `ExtendF0()` : The Hand erasing the Space.
/// The subfunction of `Extend()`.
fn ExtendF0(
    origin: usize,
    last_point: usize,
    shift: isize,
    f0_candidates: &[Vec<f64>],
    number_of_candidates: usize,
    allowed_range: f64,
    extended_f0: &mut [f64],
) -> usize {
    let threshold = 4;
    let mut tmp_f0: f64 = extended_f0[origin];
    let mut shifted_origin = origin;

    let distance = if last_point >= origin {
        last_point - origin
    } else {
        origin - last_point
    };
    let mut index_list: Vec<usize> = vec![0; distance + 1];
    for i in 0..=distance {
        index_list[i] = if shift > 0 {
            origin + shift as usize * i
        } else {
            origin - shift.abs() as usize * i
        };
    }

    let mut count = 0;
    let mut dammy = 0.0;
    for i in 0..=distance {
        let shifted_index = if shift > 0 {
            index_list[i] + shift as usize
        } else {
            index_list[i] - shift.abs() as usize
        };
        extended_f0[shifted_index] = SelectBestF0(
            tmp_f0,
            &f0_candidates[shifted_index],
            number_of_candidates,
            allowed_range,
            &mut dammy,
        );
        if extended_f0[shifted_index] == 0.0 {
            count += 1;
        } else {
            tmp_f0 = extended_f0[shifted_index];
            count = 0;
            shifted_origin = shifted_index;
        }
        if count == threshold {
            break;
        }
    }

    return shifted_origin;
}

/// Swap the f0 contour and boundary.
/// It is used in `ExtendSub()` and `MergeF0()`;
fn Swap(index1: usize, index2: usize, f0: &mut [Vec<f64>], boundary: &mut [usize]) {
    f0.swap(index1, index2);
    boundary.swap(index1 * 2, index2 * 2);
    boundary.swap(index1 * 2 + 1, index2 * 2 + 1);
}

fn ExtendSub(
    extended_f0: &mut [Vec<f64>],
    boundary_list: &mut [usize],
    number_of_sections: usize,
    // selected_extended_f0: &mut [Vec<f64>],
    // selected_boundary_list: &mut [usize],
) -> usize {
    let threshold = 2200.0;
    let mut count = 0;
    let mut mean_f0 = 0.0;
    for i in 0..number_of_sections {
        let st = boundary_list[i * 2];
        let ed = boundary_list[i * 2 + 1];
        for j in st..ed {
            mean_f0 += extended_f0[i][j];
        }
        mean_f0 /= ed as f64 - st as f64;
        if threshold / mean_f0 < ed as f64 - st as f64 {
            Swap(count, i, extended_f0, boundary_list); // extended_f0 as selected_extended_f0, boundary_list as selected_boundary_list
            count += 1;
        }
    }
    return count;
}

/// `Extend()` : The Hand erasing the Space.
fn Extend(
    multi_channel_f0: &mut [Vec<f64>],
    number_of_sections: usize,
    f0_length: usize,
    boundary_list: &mut [usize],
    f0_candidates: &[Vec<f64>],
    number_of_candidates: usize,
    allowed_range: f64,
    // extended_f0: &mut [Vec<f64>],
    // shifted_boundary_list: &mut [usize],
) -> usize {
    let threshold = 100;
    for i in 0..number_of_sections {
        // boundary_list as shifted_boundary_list
        boundary_list[i * 2 + 1] = ExtendF0(
            boundary_list[i * 2 + 1],
            MyMin(f0_length - 2, boundary_list[i * 2 + 1] + threshold),
            1,
            f0_candidates,
            number_of_candidates,
            allowed_range,
            &mut multi_channel_f0[i], // multi_channel_f0 as extended_f0
        );
        // boundary_list as shifted_boundary_list
        boundary_list[i * 2] = ExtendF0(
            boundary_list[i * 2],
            MyMax(1, boundary_list[i * 2] as isize - threshold as isize) as usize,
            -1,
            f0_candidates,
            number_of_candidates,
            allowed_range,
            &mut multi_channel_f0[i], // multi_channel_f0 as extended_f0
        );
    }

    return ExtendSub(
        multi_channel_f0,
        boundary_list, // boundary_list as shifted_boundary_list
        number_of_sections,
        // multi_channel_f0,
        // shifted_boundary_list,
    );
}

/// Indices are sorted.
fn MakeSortedOrder(boundary_list: &[usize], number_of_sections: usize, order: &mut [usize]) {
    for i in 0..number_of_sections {
        order[i] = i;
    }
    let mut tmp;
    for i in 1..number_of_sections {
        for j in (0..i).rev() {
            if boundary_list[order[j] * 2] > boundary_list[order[i] * 2] {
                tmp = order[i];
                order[i] = order[j];
                order[j] = tmp;
            } else {
                break;
            }
        }
    }
}

/// Serach the highest score with the candidate F0.
fn SearchScore(
    f0: f64,
    f0_candidates: &[f64],
    f0_scores: &[f64],
    number_of_candidates: usize,
) -> f64 {
    let mut score = 0.0;
    for i in 0..number_of_candidates {
        if f0 == f0_candidates[i] && score < f0_scores[i] {
            score = f0_scores[i];
        }
    }
    return score;
}

/// Subfunction of `MergeF0()`
fn MergeF0Sub(
    f0_1: &mut [f64],
    _f0_length: usize,
    st1: usize,
    ed1: usize,
    f0_2: &[f64],
    st2: usize,
    ed2: usize,
    f0_candidates: &[Vec<f64>],
    f0_scores: &[Vec<f64>],
    number_of_candidates: usize,
    // merged_f0: &mut [f64],
) -> usize {
    if st1 <= st2 && ed1 >= ed2 {
        return ed1;
    }

    let mut score1 = 0.0;
    let mut score2 = 0.0;
    for i in st2..=ed1 {
        score1 += SearchScore(
            f0_1[i],
            &f0_candidates[i],
            &f0_scores[i],
            number_of_candidates,
        );
        score2 += SearchScore(
            f0_2[i],
            &f0_candidates[i],
            &f0_scores[i],
            number_of_candidates,
        );
    }
    if score1 > score2 {
        for i in ed1..=ed2 {
            f0_1[i] = f0_2[i]; // f0_1 as merged_f0
        }
    } else {
        for i in st2..=ed2 {
            f0_1[i] = f0_2[i]; // f0_1 as merged_f0
        }
    }

    return ed2;
}

/// Overlapped F0 contours are merged by the likability score.
fn MergeF0(
    multi_channel_f0: &[Vec<f64>],
    boundary_list: &mut [usize],
    number_of_channels: usize,
    f0_length: usize,
    f0_candidates: &[Vec<f64>],
    f0_scores: &[Vec<f64>],
    number_of_candidates: usize,
    merged_f0: &mut [f64],
) {
    let mut order: Vec<usize> = vec![0; number_of_channels];
    MakeSortedOrder(boundary_list, number_of_channels, &mut order);

    for i in 0..f0_length {
        merged_f0[i] = multi_channel_f0[0][i];
    }

    for i in 1..number_of_channels {
        if boundary_list[order[i] * 2] as isize - boundary_list[1] as isize > 0 {
            for j in boundary_list[order[i] * 2]..=boundary_list[order[i] * 2 + 1] {
                merged_f0[j] = multi_channel_f0[order[i]][j];
            }
            boundary_list[0] = boundary_list[order[i] * 2];
            boundary_list[1] = boundary_list[order[i] * 2 + 1];
        } else {
            boundary_list[1] = MergeF0Sub(
                merged_f0,
                f0_length,
                boundary_list[0],
                boundary_list[1],
                &multi_channel_f0[order[i]],
                boundary_list[order[i] * 2],
                boundary_list[order[i] * 2 + 1],
                f0_candidates,
                f0_scores,
                number_of_candidates,
                //merged_f0,
            );
        }
    }
}

/// Step 3: Voiced sections are extended based on the continuity of F0 contour
fn FixStep3(
    f0_step2: &[f64],
    f0_length: usize,
    number_of_candidates: usize,
    f0_candidates: &[Vec<f64>],
    allowed_range: f64,
    f0_scores: &[Vec<f64>],
    f0_step3: &mut [f64],
) {
    for i in 0..f0_length {
        f0_step3[i] = f0_step2[i];
    }
    let mut boundary_list: Vec<usize> = vec![0; f0_length];
    let number_of_boundaries = GetBoundaryList(f0_step2, f0_length, &mut boundary_list);

    let mut multi_channel_f0: Vec<Vec<f64>> = vec![vec![0.0; f0_length]; number_of_boundaries / 2];
    GetMultiChannelF0(
        f0_step2,
        f0_length,
        &boundary_list,
        number_of_boundaries,
        &mut multi_channel_f0,
    );

    let number_of_channels = Extend(
        &mut multi_channel_f0,
        number_of_boundaries / 2,
        f0_length,
        &mut boundary_list,
        f0_candidates,
        number_of_candidates,
        allowed_range,
        // &mut multi_channel_f0,
        // &mut boundary_list,
    );

    if number_of_channels != 0 {
        MergeF0(
            &multi_channel_f0,
            &mut boundary_list,
            number_of_channels,
            f0_length,
            f0_candidates,
            f0_scores,
            number_of_candidates,
            f0_step3,
        );
    }
}

/// Step 4: F0s in short unvoiced section are faked
fn FixStep4(f0_step3: &[f64], f0_length: usize, threshold: usize, f0_step4: &mut [f64]) {
    for i in 0..f0_length {
        f0_step4[i] = f0_step3[i];
    }
    let mut boundary_list: Vec<usize> = vec![0; f0_length];
    let number_of_boundaries = GetBoundaryList(f0_step3, f0_length, &mut boundary_list);

    let mut distance;
    let (mut tmp0, mut tmp1, mut coefficient);
    let mut count;
    let len = if number_of_boundaries / 2 > 0 {
        number_of_boundaries / 2 - 1
    } else {
        0
    };
    for i in 0..len {
        distance = boundary_list[(i + 1) * 2] - boundary_list[i * 2 + 1] - 1;
        if distance >= threshold {
            continue;
        }
        tmp0 = f0_step3[boundary_list[i * 2 + 1]] + 1.0;
        tmp1 = f0_step3[boundary_list[(i + 1) * 2]] - 1.0;
        coefficient = (tmp1 - tmp0) / (distance as f64 + 1.0);
        count = 1;
        for j in boundary_list[i * 2 + 1] + 1..=boundary_list[(i + 1) * 2] - 1 {
            f0_step4[j] = tmp0 + coefficient * count as f64;
            count += 1;
        }
    }
}

/// `FixF0Contour()` obtains the likely F0 contour.
fn FixF0Contour(
    f0_candidates: &[Vec<f64>],
    f0_scores: &[Vec<f64>],
    f0_length: usize,
    number_of_candidates: usize,
    best_f0_contour: &mut [f64],
) {
    let mut tmp_f0_contour1: Vec<f64> = vec![0.0; f0_length];
    let mut tmp_f0_contour2: Vec<f64> = vec![0.0; f0_length];

    // These parameters are optimized by speech databases.
    SearchF0Base(
        f0_candidates,
        f0_scores,
        f0_length,
        number_of_candidates,
        &mut tmp_f0_contour1,
    );
    FixStep1(&tmp_f0_contour1, f0_length, 0.008, &mut tmp_f0_contour2);
    FixStep2(&tmp_f0_contour2, f0_length, 6, &mut tmp_f0_contour1);
    FixStep3(
        &tmp_f0_contour1,
        f0_length,
        number_of_candidates,
        f0_candidates,
        0.18,
        f0_scores,
        &mut tmp_f0_contour2,
    );
    FixStep4(&tmp_f0_contour2, f0_length, 9, best_f0_contour);
}

/// This function uses zero-lag Butterworth filter.
fn FilteringF0(
    a: &[f64],
    b: &[f64],
    x: &mut [f64],
    x_length: usize,
    st: usize,
    ed: usize,
    y: &mut [f64],
) {
    let mut w: [f64; 2] = [0.0, 0.0];
    let mut wt: f64;
    let mut tmp_x: Vec<f64> = vec![0.0; x_length];

    for i in 0..st {
        x[i] = x[st];
    }
    for i in ed + 1..x_length {
        x[i] = x[ed];
    }

    for i in 0..x_length {
        wt = x[i] + a[0] * w[0] + a[1] * w[1];
        tmp_x[x_length - i - 1] = b[0] * wt + b[1] * w[0] + b[0] * w[1];
        w[1] = w[0];
        w[0] = wt;
    }

    w[0] = 0.0;
    w[1] = 0.0;
    for i in 0..x_length {
        wt = tmp_x[i] + a[0] * w[0] + a[1] * w[1];
        y[x_length - i - 1] = b[0] * wt + b[1] * w[0] + b[0] * w[1];
        w[1] = w[0];
        w[0] = wt;
    }
}

/// `SmoothF0Contour()` uses the zero-lag Butterworth filter for smoothing.
fn SmoothF0Contour(f0: &[f64], f0_length: usize, smoothed_f0: &mut [f64]) {
    const B: [f64; 2] = [0.0078202080334971724, 0.015640416066994345];
    const A: [f64; 2] = [1.7347257688092754, -0.76600660094326412];
    let lag = 300;
    let new_f0_length = f0_length + lag * 2;
    let mut f0_contour: Vec<f64> = vec![0.0; new_f0_length];
    for i in 0..lag {
        f0_contour[i] = 0.0;
    }
    for i in lag..lag + f0_length {
        f0_contour[i] = f0[i - lag];
    }
    for i in lag + f0_length..new_f0_length {
        f0_contour[i] = 0.0;
    }

    let mut boundary_list: Vec<usize> = vec![0; new_f0_length];
    let number_of_boundaries = GetBoundaryList(&f0_contour, new_f0_length, &mut boundary_list);
    let mut multi_channel_f0: Vec<Vec<f64>> =
        vec![vec![0.0; new_f0_length]; number_of_boundaries / 2];
    GetMultiChannelF0(
        &f0_contour,
        new_f0_length,
        &boundary_list,
        number_of_boundaries,
        &mut multi_channel_f0,
    );

    for i in 0..number_of_boundaries / 2 {
        FilteringF0(
            &A,
            &B,
            &mut multi_channel_f0[i],
            new_f0_length,
            boundary_list[i * 2],
            boundary_list[i * 2 + 1],
            &mut f0_contour,
        );
        for j in boundary_list[i * 2]..=boundary_list[i * 2 + 1] {
            smoothed_f0[j - lag] = f0_contour[j];
        }
    }
}

/// `HarvestGeneralBodySub()` is the subfunction of `HarvestGeneralBody()`
fn HarvestGeneralBodySub(
    boundary_f0_list: &[f64],
    number_of_channels: usize,
    f0_length: usize,
    actual_fs: f64,
    y_length: usize,
    temporal_positions: &[f64],
    y_spectrum: &[fft_complex],
    fft_size: usize,
    f0_floor: f64,
    f0_ceil: f64,
    max_candidates: usize,
    f0_candidates: &mut [Vec<f64>],
) -> usize {
    let mut raw_f0_candidates: Vec<Vec<f64>> = vec![vec![0.0; f0_length]; number_of_channels];

    GetRawF0Candidates(
        boundary_f0_list,
        number_of_channels,
        actual_fs,
        y_length,
        temporal_positions,
        f0_length,
        y_spectrum,
        fft_size,
        f0_floor,
        f0_ceil,
        &mut raw_f0_candidates,
    );

    let number_of_candidates = DetectOfficialF0Candidates(
        &raw_f0_candidates,
        number_of_channels,
        f0_length,
        max_candidates,
        f0_candidates,
    );

    OverlapF0Candidates(f0_length, number_of_candidates, f0_candidates);

    return number_of_candidates;
}

/// `HarvestGeneralBody()` estimates the F0 contour based on Harvest.
fn HarvestGeneralBody(
    x: &[f64],
    x_length: usize,
    fs: i32,
    frame_period: i32,
    f0_floor: f64,
    f0_ceil: f64,
    channels_in_octave: f64,
    speed: usize,
    temporal_positions: &mut [f64],
    f0: &mut [f64],
) {
    let adjusted_f0_floor = f0_floor * 0.9;
    let adjusted_f0_ceil = f0_ceil * 1.1;
    let number_of_channels = (1
        + ((adjusted_f0_ceil / adjusted_f0_floor).ln() / world::kLog2 * channels_in_octave) as i32)
        as usize;
    let mut boundary_f0_list: Vec<f64> = vec![0.0; number_of_channels];
    for i in 0..number_of_channels {
        boundary_f0_list[i] =
            adjusted_f0_floor * 2.0_f64.powf((i as f64 + 1.0) / channels_in_octave);
    }

    // normalization
    let decimation_ratio = MyMax(MyMin(speed, 12), 1);
    let y_length = ((x_length as f64 / decimation_ratio as f64).ceil()) as usize;
    let actual_fs = fs as f64 / decimation_ratio as f64;
    let fft_size =
        GetSuitableFFTSize(y_length + 5 + 2 * (2.0 * actual_fs / boundary_f0_list[0]) as usize);

    // Calculation of the spectrum used for the f0 estimation
    let mut y: Vec<f64> = vec![0.0; fft_size];
    let mut y_spectrum: Vec<fft_complex> = vec![fft_complex::default(); fft_size];
    GetWaveformAndSpectrum(
        x,
        x_length,
        y_length,
        actual_fs,
        fft_size,
        decimation_ratio,
        &mut y,
        &mut y_spectrum,
    );

    let f0_length = GetSamplesForHarvest(fs, x_length, frame_period as f64);
    for i in 0..f0_length {
        temporal_positions[i] = i as f64 * frame_period as f64 / 1000.0;
        f0[i] = 0.0;
    }

    let overlap_parameter = 7;
    let max_candidates =
        (matlab_round(number_of_channels as f64 / 10.0) * overlap_parameter) as usize;
    let mut f0_candidates: Vec<Vec<f64>> = vec![vec![0.0; max_candidates]; f0_length];
    let mut f0_candidates_score: Vec<Vec<f64>> = vec![vec![0.0; max_candidates]; f0_length];

    let number_of_candidates = HarvestGeneralBodySub(
        &boundary_f0_list,
        number_of_channels,
        f0_length,
        actual_fs,
        y_length,
        temporal_positions,
        &y_spectrum,
        fft_size,
        f0_floor,
        f0_ceil,
        max_candidates,
        &mut f0_candidates,
    ) * overlap_parameter as usize;

    RefineF0Candidates(
        &y,
        y_length,
        actual_fs,
        temporal_positions,
        f0_length,
        number_of_candidates,
        f0_floor,
        f0_ceil,
        &mut f0_candidates,
        &mut f0_candidates_score,
    );
    RemoveUnreliableCandidates(
        f0_length,
        number_of_candidates,
        &mut f0_candidates,
        &mut f0_candidates_score,
    );

    let mut best_f0_contour: Vec<f64> = vec![0.0; f0_length];
    FixF0Contour(
        &f0_candidates,
        &f0_candidates_score,
        f0_length,
        number_of_candidates,
        &mut best_f0_contour,
    );
    SmoothF0Contour(&best_f0_contour, f0_length, f0);
}

//---

/// Struct for Harvest
pub struct HarvestOption {
    f0_floor: f64,
    f0_ceil: f64,
    frame_period: f64,
}

/// Harvest
///
/// - Input
///     - `x`                   : Input signal
///     - `x_length`            : Length of `x`
///     - `fs`                  : Sampling frequency
///     - `option`              : Struct to order the parameter for Harvest
///
/// - Output
///     - `temporal_positions`  : Temporal positions.
///     - `f0`                  : F0 contour.
pub fn Harvest(
    x: &[f64],
    x_length: usize,
    fs: i32,
    option: &HarvestOption,
    temporal_positions: &mut [f64],
    f0: &mut [f64],
) {
    // Several parameters will be controllable for debug.
    let target_fs = 8000.0;
    let dimension_ratio = matlab_round(fs as f64 / target_fs) as usize;
    let channels_in_octave = 40.0;

    if option.frame_period == 1.0 {
        HarvestGeneralBody(
            x,
            x_length,
            fs,
            1,
            option.f0_floor,
            option.f0_ceil,
            channels_in_octave,
            dimension_ratio,
            temporal_positions,
            f0,
        );
        return;
    }

    let basic_frame_period = 1.0;
    let basic_f0_length = GetSamplesForHarvest(fs, x_length, basic_frame_period);
    let mut basic_f0: Vec<f64> = vec![0.0; basic_f0_length];
    let mut basic_temporal_positions: Vec<f64> = vec![0.0; basic_f0_length];
    HarvestGeneralBody(
        x,
        x_length,
        fs,
        basic_frame_period as i32,
        option.f0_floor,
        option.f0_ceil,
        channels_in_octave,
        dimension_ratio,
        &mut basic_temporal_positions,
        &mut basic_f0,
    );

    let f0_length = GetSamplesForHarvest(fs, x_length, option.frame_period);
    for i in 0..f0_length {
        temporal_positions[i] = i as f64 * option.frame_period / 1000.0;
        f0[i] = basic_f0[MyMin(
            basic_f0_length - 1,
            matlab_round(temporal_positions[i] * 1000.0) as usize,
        )];
    }
}

impl HarvestOption {
    /// Set the default parameters.
    ///
    /// - Output
    ///     - Struct for the optional parameter.
    ///
    /// You can change default parameters.
    pub fn new() -> Self {
        Self {
            f0_ceil: world::kCeilF0,
            f0_floor: world::kFloorF0,
            frame_period: 5.0,
        }
    }

    pub fn f0_floor(self, f0_floor: f64) -> Self {
        Self { f0_floor, ..self }
    }

    pub fn f0_ceil(self, f0_ceil: f64) -> Self {
        Self { f0_ceil, ..self }
    }

    pub fn frame_period(self, frame_period: f64) -> Self {
        Self {
            frame_period,
            ..self
        }
    }
}

/// `GetSamplesForHarvest()` calculates the number of samples required for
/// `Harvest()`.
///
/// - Input
///     - `fs`              : Sampling frequency [Hz]
///     - `x_length`        : Length of the input signal [Sample]
///     - `frame_period`    : Frame shift [msec]
///
/// - Output
///     - The number of samples required to store the results of `Harvest()`.
pub fn GetSamplesForHarvest(fs: i32, x_length: usize, frame_period: f64) -> usize {
    return (1000.0 * x_length as f64 / fs as f64 / frame_period) as usize + 1;
}
