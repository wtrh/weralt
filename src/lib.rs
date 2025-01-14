//! Weralt is a port of [WORLD vocoder](https://github.com/mmorise/World).
//!
//! ## Example
//!
//! ```ignore
//! use weralt::{cheaptrick, d4c, harvest, synthesis, Randn};
//!
//! let frame_period = 5.; // msec
//! let fs = 44100; // sample frequency
//! let x = vec![0.; 1234]; // input signal
//! let x_length = x.len();
//!
//! let h_option = harvest::HarvestOption::new();
//! let f0_length = harvest::GetSamplesForHarvest(fs, x_length, frame_period);
//! let mut f0 = vec![0.; f0_length];
//! let mut temporal_positions = vec![0.; f0_length];
//!
//! harvest::Harvest(
//!     &x,
//!     x_length,
//!     fs,
//!     &h_option,
//!     &mut temporal_positions,
//!     &mut f0,
//! );
//!
//! let c_option = cheaptrick::CheapTrickOption::new(fs);
//! let fft_size = c_option.GetFFTSizeForCheapTrick(fs);
//! let mut spectrogram = vec![vec![0.; fft_size / 2 + 1]; f0_length];
//!
//! cheaptrick::CheapTrick::<Randn>(
//!     &x,
//!     x_length,
//!     fs,
//!     &temporal_positions,
//!     &f0,
//!     f0_length,
//!     &c_option,
//!     &mut spectrogram,
//! );
//!
//! let d4c_option = d4c::D4COption::new();
//! let mut aperiodicity = vec![vec![0.; fft_size / 2 + 1]; f0_length];
//!
//! d4c::D4C::<Randn>(
//!     &x,
//!     x_length,
//!     fs,
//!     &temporal_positions,
//!     &f0,
//!     f0_length,
//!     fft_size,
//!     &d4c_option,
//!     &mut aperiodicity,
//! );
//!
//! let y_length = ((f0_length - 1) as f64 * frame_period / 1000.0 * fs as f64) as usize + 1;
//! let mut y = vec![0.; y_length];
//!
//! synthesis::Synthesis::<Randn>( // you can get an audio signal from `y`
//!     &f0,
//!     f0_length,
//!     &spectrogram,
//!     &aperiodicity,
//!     fft_size,
//!     frame_period,
//!     fs,
//!     y_length,
//!     &mut y,
//! );
//! ```
//!
#![allow(non_snake_case)]
pub mod cheaptrick;
pub mod codec;
pub mod d4c;
pub mod dio;
pub mod harvest;
pub mod stonemask;
pub mod synthesis;

mod common;
mod constantnumbers;
mod fft;
mod matlabfunctions;

pub use matlabfunctions::{Rng, Randn};
