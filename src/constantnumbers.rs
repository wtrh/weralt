//! This module only defines constant numbers used for several function.
#![allow(non_upper_case_globals)]

pub mod world {
    // for Dio()
    pub const kCutOff: f64 = 50.0;

    //for StoneMask()
    pub const kFloorF0StoneMask: f64 = 40.0;

    pub const kPi: f64 = 3.1415926535897932384;
    pub const kMySafeGuardMinimum: f64 = 0.000000000001;
    pub const kEps: f64 = 0.00000000000000022204460492503131;
    pub const kFloorF0: f64 = 71.0;
    pub const kCeilF0: f64 = 800.0;
    pub const kDefaultF0: f64 = 500.0;
    pub const kLog2: f64 = 0.69314718055994529;
    // Maximum standard deviation not to be selected as a best f0.
    pub const kMaximumValue: f64 = 100000.0;

    // Note to me (fs: 48000)
    // 71 Hz is the limit to maintain the FFT size at 2048.
    // IF we use 70 Hz as FLOOR_F0, the FFT size of 4096 is required.

    // for D4C()
    pub const kHanning: i32 = 1;
    pub const kBlackman: i32 = 2;
    pub const kFrequencyInterval: f64 = 3000.0;
    pub const kUpperLimit: f64 = 15000.0;
    pub const kThreshold: f64 = 0.85;
    pub const kFloorF0D4C: f64 = 47.0;

    // for Codec (Mel scale)
    // S. Stevens & J. Volkmann,
    // The Relation of Pitch to Frequency: A Revised Scale,
    // American Journal of Psychology, vol. 53, no. 3, pp. 329-353, 1940.
    pub const kM0: f64 = 1127.01048;
    pub const kF0: f64 = 700.0;
    pub const kFloorFrequency: f64 = 40.0;
    pub const kCeilFrequency: f64 = 20000.0;
}
