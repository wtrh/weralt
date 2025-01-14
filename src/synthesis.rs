//! Voice synthesis based on f0, spectrogram and aperiodicity.
//! forward_real_fft, inverse_real_fft and minimum_phase are used to speed up.

use crate::common::{
    FFTProsess, ForwardRealFFT, GetSafeAperiodicity, InverseRealFFT, MinimumPhaseAnalysis, MyMin,
};
use crate::constantnumbers::world;
use crate::matlabfunctions::{fftshift, interp1, Rng};

fn GetNoiseSpectrum<R: Rng>(
    noise_size: usize,
    fft_size: usize,
    forward_real_fft: &mut ForwardRealFFT,
    randn: &mut R,
) {
    let mut average = 0.0;
    for i in 0..noise_size {
        forward_real_fft.waveform[i] = randn.gen();
        average += forward_real_fft.waveform[i];
    }

    average /= noise_size as f64;
    for i in 0..noise_size {
        forward_real_fft.waveform[i] -= average;
    }
    for i in noise_size..fft_size {
        forward_real_fft.waveform[i] = 0.0;
    }
    forward_real_fft.exec();
}

/// `GetAperiodicResponse()` calculates an aperiodic response.
fn GetAperiodicResponse<R: Rng>(
    noise_size: usize,
    fft_size: usize,
    spectrum: &[f64],
    aperiodic_ratio: &[f64],
    current_vuv: f64,
    forward_real_fft: &mut ForwardRealFFT,
    inverse_real_fft: &mut InverseRealFFT,
    minimum_phase: &mut MinimumPhaseAnalysis,
    aperiodic_response: &mut [f64],
    randn: &mut R,
) {
    GetNoiseSpectrum(noise_size, fft_size, forward_real_fft, randn);

    if current_vuv != 0.0 {
        for i in 0..=minimum_phase.fft_size / 2 {
            minimum_phase.log_spectrum[i] = (spectrum[i] * aperiodic_ratio[i]).ln() / 2.0;
        }
    } else {
        for i in 0..=minimum_phase.fft_size / 2 {
            minimum_phase.log_spectrum[i] = spectrum[i].ln() / 2.0;
        }
    }
    minimum_phase.get_minimum_phase_spectrum();

    for i in 0..=fft_size / 2 {
        inverse_real_fft.spectrum[i][0] = minimum_phase.minimum_phase_spectrum[i][0]
            * forward_real_fft.spectrum[i][0]
            - minimum_phase.minimum_phase_spectrum[i][1] * forward_real_fft.spectrum[i][1];
        inverse_real_fft.spectrum[i][1] = minimum_phase.minimum_phase_spectrum[i][0]
            * forward_real_fft.spectrum[i][1]
            + minimum_phase.minimum_phase_spectrum[i][1] * forward_real_fft.spectrum[i][0];
    }
    inverse_real_fft.exec();
    fftshift(&inverse_real_fft.waveform, fft_size, aperiodic_response);
}

/// `RemoveDCComponent()`
fn RemoveDCComponent(periodic_response: &mut [f64], fft_size: usize, dc_remover: &[f64]) {
    let mut dc_component = 0.0;
    for i in fft_size / 2..fft_size {
        dc_component += periodic_response[i];
    }
    for i in 0..fft_size / 2 {
        periodic_response[i] = -dc_component * dc_remover[i];
    }
    for i in fft_size / 2..fft_size {
        periodic_response[i] -= dc_component * dc_remover[i];
    }
}

/// `GetSpectrumWithFractionalTimeShift()` calculates a periodic spectrum with
/// the fractional time shift under 1/fs.
fn GetSpectrumWithFractionalTimeShift(
    fft_size: usize,
    coefficient: f64,
    inverse_real_fft: &mut InverseRealFFT,
) {
    let (mut re, mut im, mut re2, mut im2);
    for i in 0..=fft_size / 2 {
        re = inverse_real_fft.spectrum[i][0];
        im = inverse_real_fft.spectrum[i][1];
        re2 = (coefficient * i as f64).cos();
        im2 = (1.0 - re2 * re2).sqrt(); // sin(pshift)

        inverse_real_fft.spectrum[i][0] = re * re2 + im * im2;
        inverse_real_fft.spectrum[i][1] = im * re2 - re * im2;
    }
}

/// `GetPeriodicResponse()` calculates a periodic response.
fn GetPeriodicResponse(
    fft_size: usize,
    spectrum: &[f64],
    aperiodic_ratio: &[f64],
    current_vuv: f64,
    inverse_real_fft: &mut InverseRealFFT,
    minimum_phase: &mut MinimumPhaseAnalysis,
    dc_remover: &[f64],
    fractional_time_shift: f64,
    fs: i32,
    periodic_response: &mut [f64],
) {
    if current_vuv <= 0.5 || aperiodic_ratio[0] > 0.999 {
        for i in 0..fft_size {
            periodic_response[i] = 0.0;
        }
        return;
    }

    for i in 0..=minimum_phase.fft_size / 2 {
        minimum_phase.log_spectrum[i] =
            (spectrum[i] * (1.0 - aperiodic_ratio[i]) + world::kMySafeGuardMinimum).ln() / 2.0;
    }
    minimum_phase.get_minimum_phase_spectrum();

    for i in 0..=fft_size / 2 {
        inverse_real_fft.spectrum[i][0] = minimum_phase.minimum_phase_spectrum[i][0];
        inverse_real_fft.spectrum[i][1] = minimum_phase.minimum_phase_spectrum[i][1];
    }

    // apply fractional time delay of fractional_time_shift seconds
    // using linear phase shift
    let coefficient = 2.0 * world::kPi * fractional_time_shift * fs as f64 / fft_size as f64;
    GetSpectrumWithFractionalTimeShift(fft_size, coefficient, inverse_real_fft);

    inverse_real_fft.exec();
    fftshift(&inverse_real_fft.waveform, fft_size, periodic_response);
    RemoveDCComponent(periodic_response, fft_size, dc_remover);
}

fn GetSpectralEnvelope(
    current_time: f64,
    frame_period: f64,
    f0_length: usize,
    spectrogram: &[Vec<f64>],
    fft_size: usize,
    spectral_envelope: &mut [f64],
) {
    let current_frame_floor = MyMin(
        f0_length - 1,
        (current_time / frame_period).floor() as usize,
    );
    let current_frame_ceil = MyMin(f0_length - 1, (current_time / frame_period).ceil() as usize);
    let interpolation = current_time / frame_period - current_frame_floor as f64;

    if current_frame_floor == current_frame_ceil {
        for i in 0..=fft_size / 2 {
            spectral_envelope[i] = spectrogram[current_frame_floor][i].abs();
        }
    } else {
        for i in 0..=fft_size / 2 {
            spectral_envelope[i] = (1.0 - interpolation)
                * spectrogram[current_frame_floor][i].abs()
                + interpolation * spectrogram[current_frame_ceil][i].abs();
        }
    }
}

fn GetAperiodicRatio(
    current_time: f64,
    frame_period: f64,
    f0_length: usize,
    aperiodicity: &[Vec<f64>],
    fft_size: usize,
    aperiodic_spectrum: &mut [f64],
) {
    let current_frame_floor = MyMin(
        f0_length - 1,
        (current_time / frame_period).floor() as usize,
    );
    let current_frame_ceil = MyMin(f0_length - 1, (current_time / frame_period).ceil() as usize);
    let interpolation = current_time / frame_period - current_frame_floor as f64;

    if current_frame_floor == current_frame_ceil {
        for i in 0..=fft_size / 2 {
            aperiodic_spectrum[i] =
                GetSafeAperiodicity(aperiodicity[current_frame_floor][i]).powf(2.0);
        }
    } else {
        for i in 0..=fft_size / 2 {
            aperiodic_spectrum[i] = ((1.0 - interpolation)
                * GetSafeAperiodicity(aperiodicity[current_frame_floor][i])
                + interpolation * GetSafeAperiodicity(aperiodicity[current_frame_ceil][i]))
            .powf(2.0);
        }
    }
}

/// `GetOneFrameSegment()` calculates a periodic and aperiodic response at a time.
fn GetOneFrameSegment<R: Rng>(
    current_vuv: f64,
    noise_size: usize,
    spectrogram: &[Vec<f64>],
    fft_size: usize,
    aperiodicity: &[Vec<f64>],
    f0_length: usize,
    frame_period: f64,
    current_time: f64,
    fractional_time_shift: f64,
    fs: i32,
    forward_real_fft: &mut ForwardRealFFT,
    inverse_real_fft: &mut InverseRealFFT,
    minimum_phase: &mut MinimumPhaseAnalysis,
    dc_remover: &[f64],
    response: &mut [f64],
    randn: &mut R,
) {
    let mut aperiodic_response = vec![0.0; fft_size];
    let mut periodic_response = vec![0.0; fft_size];

    let mut spectral_envelope = vec![0.0; fft_size];
    let mut aperiodic_ratio = vec![0.0; fft_size];
    GetSpectralEnvelope(
        current_time,
        frame_period,
        f0_length,
        spectrogram,
        fft_size,
        &mut spectral_envelope,
    );
    GetAperiodicRatio(
        current_time,
        frame_period,
        f0_length,
        aperiodicity,
        fft_size,
        &mut aperiodic_ratio,
    );

    // Synthesis of the periodic response
    GetPeriodicResponse(
        fft_size,
        &spectral_envelope,
        &aperiodic_ratio,
        current_vuv,
        inverse_real_fft,
        minimum_phase,
        dc_remover,
        fractional_time_shift,
        fs,
        &mut periodic_response,
    );

    // Synthesis of the aperiodic response
    GetAperiodicResponse(
        noise_size,
        fft_size,
        &spectral_envelope,
        &aperiodic_ratio,
        current_vuv,
        forward_real_fft,
        inverse_real_fft,
        minimum_phase,
        &mut aperiodic_response,
        randn,
    );

    let sqrt_noise_size = (noise_size as f64).sqrt();
    for i in 0..fft_size {
        response[i] =
            (periodic_response[i] * sqrt_noise_size + aperiodic_response[i]) / fft_size as f64;
    }
}

fn GetTemporalParametersForTimeBase(
    f0: &[f64],
    f0_length: usize,
    fs: i32,
    y_length: usize,
    frame_period: f64,
    lowest_f0: f64,
    time_axis: &mut [f64],
    coarse_time_axis: &mut [f64],
    coarse_f0: &mut [f64],
    coarse_vuv: &mut [f64],
) {
    for i in 0..y_length {
        time_axis[i] = i as f64 / fs as f64;
    }
    // the array 'coarse_time_axis' is supposed to have 'f0_length + 1' positions
    for i in 0..f0_length {
        coarse_time_axis[i] = i as f64 * frame_period;
        coarse_f0[i] = if f0[i] < lowest_f0 { 0.0 } else { f0[i] };
        coarse_vuv[i] = if coarse_f0[i] == 0.0 { 0.0 } else { 1.0 };
    }
    coarse_time_axis[f0_length] = f0_length as f64 * frame_period;
    coarse_f0[f0_length] = coarse_f0[f0_length - 1] * 2.0 - coarse_f0[f0_length - 2];
    coarse_vuv[f0_length] = coarse_vuv[f0_length - 1] * 2.0 - coarse_vuv[f0_length - 2];
}

fn GetPulseLocationsForTimeBase(
    interpolated_f0: &[f64],
    time_axis: &[f64],
    y_length: usize,
    fs: i32,
    pulse_locations: &mut [f64],
    pulse_locations_index: &mut [usize],
    pulse_locations_time_shift: &mut [f64],
) -> usize {
    let mut total_phase = vec![0.0; y_length];
    let mut wrap_phase = vec![0.0; y_length];
    let mut wrap_phase_abs = vec![0.0; y_length - 1];
    total_phase[0] = 2.0 * world::kPi * interpolated_f0[0] / fs as f64;
    wrap_phase[0] = total_phase[0] % (2.0 * world::kPi);
    for i in 1..y_length {
        total_phase[i] = total_phase[i - 1] + 2.0 * world::kPi * interpolated_f0[i] / fs as f64;
        wrap_phase[i] = total_phase[i] % (2.0 * world::kPi);
        wrap_phase_abs[i - 1] = (wrap_phase[i] - wrap_phase[i - 1]).abs();
    }

    let mut number_of_pulses = 0;
    for i in 0..y_length - 1 {
        if wrap_phase_abs[i] > world::kPi {
            pulse_locations[number_of_pulses] = time_axis[i];
            pulse_locations_index[number_of_pulses] = i;

            // calculate the time shift in seconds between exact fractional pulse
            // position and the integer pulse position (sample i)
            // as we don't have access to the exact pulse position, we infer it
            // from the point between sample i and sample i + 1 where the
            // accummulated phase cross a multiple of 2pi
            // this point is found by solving y1 + x * (y2 - y1) = 0 for x, where y1
            // and y2 are the phases corresponding to sample i and i + 1, offset so
            // they cross zero; x >= 0
            let y1 = wrap_phase[i] - 2.0 * world::kPi;
            let y2 = wrap_phase[i + 1];
            let x = -y1 / (y2 - y1);
            pulse_locations_time_shift[number_of_pulses] = x / fs as f64;

            number_of_pulses += 1;
        }
    }

    return number_of_pulses;
}

fn GetTimeBase(
    f0: &[f64],
    f0_length: usize,
    fs: i32,
    frame_period: f64,
    y_length: usize,
    lowest_f0: f64,
    pulse_locations: &mut [f64],
    pulse_locations_index: &mut [usize],
    pulse_locations_time_shift: &mut [f64],
    interpolated_vuv: &mut [f64],
) -> usize {
    let mut time_axis = vec![0.0; y_length];
    let mut coarse_time_axis = vec![0.0; f0_length + 1];
    let mut coarse_f0 = vec![0.0; f0_length + 1];
    let mut coarse_vuv = vec![0.0; f0_length + 1];
    GetTemporalParametersForTimeBase(
        f0,
        f0_length,
        fs,
        y_length,
        frame_period,
        lowest_f0,
        &mut time_axis,
        &mut coarse_time_axis,
        &mut coarse_f0,
        &mut coarse_vuv,
    );
    let mut interpolated_f0 = vec![0.0; y_length];
    interp1(
        &coarse_time_axis,
        &coarse_f0,
        f0_length + 1,
        &time_axis,
        y_length,
        &mut interpolated_f0,
    );
    interp1(
        &coarse_time_axis,
        &coarse_vuv,
        f0_length + 1,
        &time_axis,
        y_length,
        interpolated_vuv,
    );

    for i in 0..y_length {
        interpolated_vuv[i] = if interpolated_vuv[i] > 0.5 { 1.0 } else { 0.0 };
        interpolated_f0[i] = if interpolated_vuv[i] == 0.0 {
            world::kDefaultF0
        } else {
            interpolated_f0[i]
        };
    }

    let number_of_pulses = GetPulseLocationsForTimeBase(
        &interpolated_f0,
        &time_axis,
        y_length,
        fs,
        pulse_locations,
        pulse_locations_index,
        pulse_locations_time_shift,
    );

    return number_of_pulses;
}

fn GetDCRemover(fft_size: usize, dc_remover: &mut [f64]) {
    let mut dc_component = 0.0;
    for i in 0..fft_size / 2 {
        dc_remover[i] =
            0.5 - 0.5 * (2.0 * world::kPi * (i as f64 + 1.0) / (1.0 + fft_size as f64)).cos();
        dc_remover[fft_size - i - 1] = dc_remover[i];
        dc_component += dc_remover[i] * 2.0;
    }
    for i in 0..fft_size / 2 {
        dc_remover[i] /= dc_component;
        dc_remover[fft_size - i - 1] = dc_remover[i];
    }
}

//---

/// `Synthesis()` synthesize the voice based on f0, spectrogram and
/// aperiodicity (not excitation signal).
///
/// - Input
///     - `f0`              : f0 contour
///     - `f0_length`       : Length of `f0`
///     - `spectrogram`     : Spectrogram estimated by CheapTrick
///     - `fft_size`        : FFT size
///     - `aperiodicity`    : Aperiodicity spectrogram based on D4C
///     - `frame_period`    : Temporal period used for the analysis
///     - `fs`              : Sampling frequency
///     - `y_length`        : Length of the output signal (Memory of y has been
///         allocated in advance)
/// - Output
///     - `y`               : Calculated speech
pub fn Synthesis<R: Rng>(
    f0: &[f64],
    f0_length: usize,
    spectrogram: &[Vec<f64>],
    aperiodicity: &[Vec<f64>],
    fft_size: usize,
    mut frame_period: f64,
    fs: i32,
    y_length: usize,
    y: &mut [f64],
) {
    let mut randn = R::new();

    let mut impulse_response = vec![0.0; fft_size];

    for i in 0..y_length {
        y[i] = 0.0;
    }

    let mut minimum_phase = MinimumPhaseAnalysis::new(fft_size);
    let mut inverse_real_fft = InverseRealFFT::new(fft_size);
    let mut forward_real_fft = ForwardRealFFT::new(fft_size);

    let mut pulse_locations = vec![0.0; y_length];
    let mut pulse_locations_index = vec![0; y_length];
    let mut pulse_locations_time_shift = vec![0.0; y_length];
    let mut interpolated_vuv = vec![0.0; y_length];
    let number_of_pulses = GetTimeBase(
        f0,
        f0_length,
        fs,
        frame_period / 1000.0,
        y_length,
        (fs as usize / fft_size) as f64 + 1.0,
        &mut pulse_locations,
        &mut pulse_locations_index,
        &mut pulse_locations_time_shift,
        &mut interpolated_vuv,
    );

    let mut dc_remover = vec![0.0; fft_size];
    GetDCRemover(fft_size, &mut dc_remover);

    frame_period /= 1000.0;
    let mut noise_size;
    let (mut index, mut offset, mut lower_limit, mut upper_limit);
    for i in 0..number_of_pulses {
        noise_size =
            pulse_locations_index[MyMin(number_of_pulses - 1, i + 1)] - pulse_locations_index[i];

        GetOneFrameSegment(
            interpolated_vuv[pulse_locations_index[i]],
            noise_size,
            spectrogram,
            fft_size,
            aperiodicity,
            f0_length,
            frame_period,
            pulse_locations[i],
            pulse_locations_time_shift[i],
            fs,
            &mut forward_real_fft,
            &mut inverse_real_fft,
            &mut minimum_phase,
            &dc_remover,
            &mut impulse_response,
            &mut randn,
        );
        if pulse_locations_index[i] as usize >= fft_size / 2 {
            offset = pulse_locations_index[i] as usize - fft_size / 2 + 1;
            lower_limit = 0;
            upper_limit = MyMin(fft_size, y_length - offset);
            for j in lower_limit..upper_limit {
                index = j + offset;
                y[index] += impulse_response[j];
            }
        } else {
            offset = fft_size / 2 - pulse_locations_index[i] as usize + 1;
            lower_limit = offset;
            upper_limit = MyMin(fft_size, y_length + offset);
            for j in lower_limit..upper_limit {
                index = j - offset;
                y[index] += impulse_response[j];
            }
        }
    }
}
