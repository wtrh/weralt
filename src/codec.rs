//! Coder/decoder functions for the spectral envelope and aperiodicity.

use crate::common::{FFTProsess, ForwardRealFFT, InverseComplexFFT, MyMin};
use crate::constantnumbers::world;
use crate::fft::fft_complex;
use crate::matlabfunctions::{interp1, interp1Q};

//---

/// Aperiodicity is initialized by the value 1.0 - world::kMySafeGuardMinimum.
/// This value means the frame/frequency index is aperiodic.
fn InitializeAperiodicity(f0_length: usize, fft_size: usize, aperiodicity: &mut [Vec<f64>]) {
    for i in 0..f0_length {
        for j in 0..fft_size / 2 + 1 {
            aperiodicity[i][j] = 1.0 - world::kMySafeGuardMinimum;
        }
    }
}

/// This function identifies whether this frame is voiced or unvoiced.
fn CheckVUV(
    coarse_aperiodicity: &[f64],
    number_of_aperiodicities: usize,
    tmp_aperiodicity: &mut [f64],
) -> i32 {
    let mut tmp = 0.0;
    for i in 0..number_of_aperiodicities {
        tmp += coarse_aperiodicity[i];
        tmp_aperiodicity[i + 1] = coarse_aperiodicity[i];
    }
    tmp /= number_of_aperiodicities as f64;

    return if tmp > -0.5 { 1 } else { 0 }; // -0.5 is not optimized, but okay.
}

/// Aperiodicity is obtained from the coded aperiodicity.
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

/// Frequency is converted into its mel representation.
#[inline]
fn FrequencyToMel(frequency: f64) -> f64 {
    return world::kM0 * (frequency / world::kF0 + 1.0).ln();
}

/// Mel is converted into frequency.
#[inline]
fn MelToFrequency(mel: f64) -> f64 {
    return world::kF0 * ((mel / world::kM0).exp() - 1.0);
}

/// DCT for spectral envelope coding
fn DCTForCodec(
    mel_spectrum: &[f64],
    max_dimension: usize,
    weight: &[fft_complex],
    forward_real_fft: &mut ForwardRealFFT,
    number_of_dimensions: usize,
    mel_cepstrum: &mut [f64],
) {
    let bias = max_dimension / 2;
    for i in 0..max_dimension / 2 {
        forward_real_fft.waveform[i] = mel_spectrum[i * 2];
        forward_real_fft.waveform[i + bias] = mel_spectrum[max_dimension - (i * 2) - 1];
    }
    forward_real_fft.exec();

    let normalization = (forward_real_fft.fft_size as f64).sqrt();
    for i in 0..number_of_dimensions {
        mel_cepstrum[i] = (forward_real_fft.spectrum[i][0] * weight[i][0]
            - forward_real_fft.spectrum[i][1] * weight[i][1])
            / normalization;
    }
}

/// IDCT for spectral envelope decoding
fn IDCTForCodec(
    mel_cepstrum: &[f64],
    max_dimension: usize,
    weight: &[fft_complex],
    inverse_complex_fft: &mut InverseComplexFFT,
    number_of_dimensions: usize,
    mel_spectrum: &mut [f64],
) {
    let normalization = (inverse_complex_fft.fft_size as f64).sqrt();
    for i in 0..number_of_dimensions {
        inverse_complex_fft.input[i][0] = mel_cepstrum[i] * weight[i][0] * normalization;
        inverse_complex_fft.input[i][1] = -mel_cepstrum[i] * weight[i][1] * normalization;
    }
    for i in number_of_dimensions..max_dimension {
        inverse_complex_fft.input[i][0] = 0.0;
        inverse_complex_fft.input[i][1] = 0.0;
    }

    inverse_complex_fft.exec();

    for i in 0..max_dimension / 2 {
        mel_spectrum[i * 2] = inverse_complex_fft.output[i][0];
        mel_spectrum[(i * 2) + 1] = inverse_complex_fft.output[max_dimension - i - 1][0];
    }
}

/// Spectral envelope in a frame is coded
fn CodeOneFrame(
    log_spectral_envelope: &[f64],
    frequency_axis: &[f64],
    fft_size: usize,
    mel_axis: &[f64],
    weight: &[fft_complex],
    max_dimension: usize,
    number_of_dimensions: usize,
    forward_real_fft: &mut ForwardRealFFT,
    coded_spectral_envelope: &mut [f64],
) {
    let mut mel_spectrum: Vec<f64> = vec![0.0; max_dimension];
    interp1(
        frequency_axis,
        log_spectral_envelope,
        fft_size / 2 + 1,
        mel_axis,
        max_dimension,
        &mut mel_spectrum,
    );

    // DCT
    DCTForCodec(
        &mel_spectrum,
        max_dimension,
        weight,
        forward_real_fft,
        number_of_dimensions,
        coded_spectral_envelope,
    );
}

/// Coded spectral envelope in a frame is decoded
fn DecodeOneFrame(
    coded_spectral_envelope: &[f64],
    frequency_axis: &[f64],
    fft_size: usize,
    mel_axis: &[f64],
    weight: &[fft_complex],
    max_dimension: usize,
    number_of_dimensions: usize,
    inverse_complex_fft: &mut InverseComplexFFT,
    spectral_envelope: &mut [f64],
) {
    let mut mel_spectrum: Vec<f64> = vec![0.0; max_dimension + 2];

    // IDCT
    IDCTForCodec(
        coded_spectral_envelope,
        max_dimension,
        weight,
        inverse_complex_fft,
        number_of_dimensions,
        &mut mel_spectrum[1..], // slice
    );
    mel_spectrum[0] = mel_spectrum[1];
    mel_spectrum[max_dimension + 1] = mel_spectrum[max_dimension];

    interp1(
        mel_axis,
        &mel_spectrum,
        max_dimension + 2,
        frequency_axis,
        fft_size / 2 + 1,
        spectral_envelope,
    );

    for i in 0..fft_size / 2 + 1 {
        spectral_envelope[i] = (spectral_envelope[i] / max_dimension as f64).exp();
    }
}

/// GetParameters() generates the required parameters.
fn GetParametersForCoding(
    floor_frequency: f64,
    ceil_frequency: f64,
    fs: i32,
    fft_size: usize,
    mel_axis: &mut [f64],
    frequency_axis: &mut [f64],
    weight: &mut [fft_complex],
) {
    let max_dimension = fft_size / 2;
    let floor_mel = FrequencyToMel(floor_frequency);
    let ceil_mel = FrequencyToMel(ceil_frequency);

    // Generate the mel axis and the weighting vector for DCT.
    for i in 0..max_dimension {
        mel_axis[i] = (ceil_mel - floor_mel) * i as f64 / max_dimension as f64 + floor_mel;
        weight[i][0] =
            2.0 * (i as f64 * world::kPi / fft_size as f64).cos() / (fft_size as f64).sqrt();
        weight[i][1] =
            2.0 * (i as f64 * world::kPi / fft_size as f64).sin() / (fft_size as f64).sqrt();
    }
    weight[0][0] /= (2.0_f64).sqrt();

    // Generate the frequency axis on mel scale
    for i in 0..=max_dimension {
        frequency_axis[i] = FrequencyToMel(i as f64 * fs as f64 / fft_size as f64);
    }
}

/// GetParameters() generates the required parameters.
fn GetParametersForDecoding(
    floor_frequency: f64,
    ceil_frequency: f64,
    fs: i32,
    fft_size: usize,
    number_of_dimensions: usize,
    mel_axis: &mut [f64],
    frequency_axis: &mut [f64],
    weight: &mut [fft_complex],
) {
    let max_dimension = fft_size / 2;
    let floor_mel = FrequencyToMel(floor_frequency);
    let ceil_mel = FrequencyToMel(ceil_frequency);

    // Generate the weighting vector for IDCT.
    for i in 0..number_of_dimensions {
        weight[i][0] = (i as f64 * world::kPi / fft_size as f64).cos() * (fft_size as f64).sqrt();
        weight[i][1] = (i as f64 * world::kPi / fft_size as f64).sin() * (fft_size as f64).sqrt();
    }
    weight[0][0] /= (2.0_f64).sqrt();
    // Generate the mel axis for IDCT.
    for i in 0..max_dimension {
        mel_axis[i + 1] =
            MelToFrequency((ceil_mel - floor_mel) * i as f64 / max_dimension as f64 + floor_mel);
    }
    mel_axis[0] = 0.0;
    mel_axis[max_dimension + 1] = fs as f64 / 2.0;

    // Generate the frequency axis
    for i in 0..fft_size / 2 + 1 {
        frequency_axis[i] = i as f64 * fs as f64 / fft_size as f64;
    }
}

//---

/// GetNumberOfAperiodicities provides the number of dimensions for aperiodicity
/// coding. It is determined by only fs.
///
/// - Input
///     - `fs`   : Sampling frequency
///
/// - Output
///     - Number of aperiodicities
pub fn GetNumberOfAperiodicities(fs: i32) -> i32 {
    return (MyMin(
        world::kUpperLimit,
        fs as f64 / 2.0 - world::kFrequencyInterval,
    ) / world::kFrequencyInterval) as i32;
}

/// CodeAperiodicity codes the aperiodicity. The number of dimensions is
/// determined by fs.
///
/// - Input
///     - `aperiodicity`        : Aperiodicity before coding
///     - `f0_length`           : Length of F0 contour
///     - `fs`                  : Sampling frequency
///     - `fft_size`            : FFT size
///
/// - Output
///     - `coded_aperiodicity`  : Coded aperiodicity
pub fn CodeAperiodicity(
    aperiodicity: &[Vec<f64>],
    f0_length: usize,
    fs: i32,
    fft_size: usize,
    coded_aperiodicity: &mut [Vec<f64>],
) {
    let number_of_aperiodicities = GetNumberOfAperiodicities(fs) as usize;
    let mut coarse_frequency_axis: Vec<f64> = vec![0.0; number_of_aperiodicities];
    for i in 0..number_of_aperiodicities {
        coarse_frequency_axis[i] = world::kFrequencyInterval * (i as f64 + 1.0);
    }

    let mut log_aperiodicity: Vec<f64> = vec![0.0; fft_size / 2 + 1];
    for i in 0..f0_length {
        for j in 0..fft_size / 2 + 1 {
            log_aperiodicity[j] = 20.0 * (aperiodicity[i][j]).log10();
        }
        interp1Q(
            0.0,
            fs as f64 / fft_size as f64,
            &log_aperiodicity,
            fft_size / 2 + 1,
            &coarse_frequency_axis,
            number_of_aperiodicities,
            &mut coded_aperiodicity[i],
        );
    }
}

/// DecodeAperiodicity decodes the coded aperiodicity.
///
/// - Input
///     - `coded_aperiodicity`  : Coded aperiodicity
///     - `f0_length`           : Length of F0 contour
///     - `fs`                  : Sampling frequency
///     - `fft_size`            : FFT size
///
/// - Output
///     - `aperiodicity`        : Decoded aperiodicity
pub fn DecodeAperiodicity(
    coded_aperiodicity: &[Vec<f64>],
    f0_length: usize,
    fs: i32,
    fft_size: usize,
    aperiodicity: &mut [Vec<f64>],
) {
    InitializeAperiodicity(f0_length, fft_size, aperiodicity);
    let number_of_aperiodicities = GetNumberOfAperiodicities(fs) as usize;
    let mut frequency_axis: Vec<f64> = vec![0.0; fft_size / 2 + 1];
    for i in 0..=fft_size / 2 {
        frequency_axis[i] = fs as f64 / fft_size as f64 * i as f64;
    }
    let mut coarse_frequency_axis: Vec<f64> = vec![0.0; number_of_aperiodicities + 2];
    for i in 0..=number_of_aperiodicities {
        coarse_frequency_axis[i] = i as f64 * world::kFrequencyInterval;
    }
    coarse_frequency_axis[number_of_aperiodicities + 1] = fs as f64 / 2.0;
    let mut coarse_aperiodicity: Vec<f64> = vec![0.0; number_of_aperiodicities + 2];
    coarse_aperiodicity[0] = -60.0;
    coarse_aperiodicity[number_of_aperiodicities + 1] = -world::kMySafeGuardMinimum;
    for i in 0..f0_length {
        if CheckVUV(
            &coded_aperiodicity[i],
            number_of_aperiodicities,
            &mut coarse_aperiodicity,
        ) == 1
        {
            continue;
        }
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

/// CodeSpectralEnvelope codes the spectral envelope.
///
/// - Input
///     - `aperiodicity`            : Aperiodicity before coding
///     - `f0_length`               : Length of F0 contour
///     - `fs`                      : Sampling frequency
///     - `fft_size`                : FFT size
///     - `number_of_dimensions`    : Parameter for compression
///
/// - Output
///     - `coded_spectral_envelope`
pub fn CodeSpectralEnvelope(
    spectrogram: &[Vec<f64>],
    f0_length: usize,
    fs: i32,
    fft_size: usize,
    number_of_dimensions: usize,
    coded_spectral_envelope: &mut [Vec<f64>],
) {
    let mut mel_axis: Vec<f64> = vec![0.0; fft_size / 2];
    let mut frequency_axis: Vec<f64> = vec![0.0; fft_size / 2 + 1];
    let mut tmp_spectrum: Vec<f64> = vec![0.0; fft_size / 2 + 1];
    let mut weight: Vec<fft_complex> = vec![fft_complex::default(); fft_size / 2];
    // Generation of the required parameters
    GetParametersForCoding(
        world::kFloorFrequency,
        MyMin(fs as f64 / 2.0, world::kCeilFrequency),
        fs,
        fft_size,
        &mut mel_axis,
        &mut frequency_axis,
        &mut weight,
    );

    let mut forward_real_fft = ForwardRealFFT::new(fft_size / 2);
    for i in 0..f0_length {
        for j in 0..fft_size / 2 + 1 {
            tmp_spectrum[j] = (spectrogram[i][j]).ln();
        }
        CodeOneFrame(
            &tmp_spectrum,
            &frequency_axis,
            fft_size,
            &mel_axis,
            &weight,
            fft_size / 2,
            number_of_dimensions,
            &mut forward_real_fft,
            &mut coded_spectral_envelope[i],
        );
    }
}

/// DecodeSpectralEnvelope decodes the coded spectral envelope.
///
/// - Input
///     - `coded_aperiodicity`      : Coded aperiodicity
///     - `f0_length`               : Length of F0 contour
///     - `fs`                      : Sampling frequency
///     - `fft_size`                : FFT size
///     - `number_of_dimensions`    : Parameter for compression
///
/// - Output
///     - `spectrogram`
pub fn DecodeSpectralEnvelope(
    coded_spectral_envelope: &[Vec<f64>],
    f0_length: usize,
    fs: i32,
    fft_size: usize,
    number_of_dimensions: usize,
    spectrogram: &mut [Vec<f64>],
) {
    let mut mel_axis: Vec<f64> = vec![0.0; fft_size / 2 + 2];
    let mut frequency_axis: Vec<f64> = vec![0.0; fft_size / 2 + 1];
    let mut weight: Vec<fft_complex> = vec![fft_complex::default(); fft_size / 2];

    // Generation of the required parameters
    GetParametersForDecoding(
        world::kFloorFrequency,
        MyMin(fs as f64 / 2.0, world::kCeilFrequency),
        fs,
        fft_size,
        number_of_dimensions,
        &mut mel_axis,
        &mut frequency_axis,
        &mut weight,
    );

    let mut inverse_complex_fft = InverseComplexFFT::new(fft_size / 2);

    for i in 0..f0_length {
        DecodeOneFrame(
            &coded_spectral_envelope[i],
            &frequency_axis,
            fft_size,
            &mel_axis,
            &weight,
            fft_size / 2,
            number_of_dimensions,
            &mut inverse_complex_fft,
            &mut spectrogram[i],
        );
    }
}
