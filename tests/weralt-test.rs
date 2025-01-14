extern crate hound;
extern crate weralt;

use core::cmp::{max, min};
use weralt::{cheaptrick, d4c, harvest, synthesis, Randn};

fn resynthesis(x: &[f64], fs: i32) -> Vec<f64> {
    let frame_period = 5.; // msec
    let x_length = x.len();

    let h_option = harvest::HarvestOption::new();
    let f0_length = harvest::GetSamplesForHarvest(fs, x_length, frame_period);
    let mut f0 = vec![0.; f0_length];
    let mut temporal_positions = vec![0.; f0_length];

    harvest::Harvest(
        &x,
        x_length,
        fs,
        &h_option,
        &mut temporal_positions,
        &mut f0,
    );

    let c_option = cheaptrick::CheapTrickOption::new(fs);
    let fft_size = c_option.GetFFTSizeForCheapTrick(fs);
    let mut spectrogram = vec![vec![0.; fft_size / 2 + 1]; f0_length];

    cheaptrick::CheapTrick::<Randn>(
        &x,
        x_length,
        fs,
        &temporal_positions,
        &f0,
        f0_length,
        &c_option,
        &mut spectrogram,
    );

    let d4c_option = d4c::D4COption::new();
    let mut aperiodicity = vec![vec![0.; fft_size / 2 + 1]; f0_length];

    d4c::D4C::<Randn>(
        &x,
        x_length,
        fs,
        &temporal_positions,
        &f0,
        f0_length,
        fft_size,
        &d4c_option,
        &mut aperiodicity,
    );

    let y_length = ((f0_length - 1) as f64 * frame_period / 1000.0 * fs as f64) as usize + 1;
    let mut y = vec![0.; y_length];

    synthesis::Synthesis::<Randn>(
        &f0,
        f0_length,
        &spectrogram,
        &aperiodicity,
        fft_size,
        frame_period,
        fs,
        y_length,
        &mut y,
    );

    return y;
}

#[test]
fn test_resynthesis() {
    let i_fname = "tests/sample/input.wav";
    let o_fname = "tests/sample/output.wav";

    let mut reader = hound::WavReader::open(i_fname).expect("Failed to load audio file.");
    let sample_rate = reader.spec().sample_rate;
    let samples = reader.samples::<i16>();

    let mut x = Vec::new();
    for s in samples {
        x.push(s.expect("The audio file is corrupted.") as f64);
    }

    let y = resynthesis(&x, sample_rate as i32);

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(o_fname, spec).expect("Failed to write audio file.");
    for i in 0..y.len() {
        writer
            .write_sample(max(i16::MIN, min(i16::MAX, y[i] as i16)))
            .unwrap();
    }
}
