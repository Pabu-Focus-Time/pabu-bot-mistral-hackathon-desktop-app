use image_compare::rgb_hybrid_compare;
use std::{env, process};

fn main() {
	let args: Vec<String> = env::args().collect();
	if args.len() != 3 {
		eprintln!("Usage: cargo run --example rgb_hybrid_score -- <image_a> <image_b>");
		process::exit(1);
	}

	let image_a_path = &args[1];
	let image_b_path = &args[2];

	let image_a = match image::open(image_a_path) {
		Ok(img) => img.into_rgb8(),
		Err(err) => {
			eprintln!("Failed to open first image '{}': {}", image_a_path, err);
			process::exit(1);
		}
	};

	let image_b = match image::open(image_b_path) {
		Ok(img) => img.into_rgb8(),
		Err(err) => {
			eprintln!("Failed to open second image '{}': {}", image_b_path, err);
			process::exit(1);
		}
	};

	let result = match rgb_hybrid_compare(&image_a, &image_b) {
		Ok(v) => v,
		Err(err) => {
			eprintln!("RGB hybrid comparison failed: {}", err);
			process::exit(1);
		}
	};

	println!("{:.6}", result.score);
}
