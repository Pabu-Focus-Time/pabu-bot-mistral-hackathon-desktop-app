use image_compare::{gray_similarity_histogram, gray_similarity_structure, rgb_hybrid_compare, Algorithm, Metric};
use std::{env, process};

fn usage() {
	eprintln!("Usage:");
	eprintln!("  cargo run --example demo_compare -- <image_a> <image_b> [threshold] [diff_output]");
	eprintln!();
	eprintln!("Example:");
	eprintln!("  cargo run --example demo_compare -- ../scripts/image_3.png ../scripts/image_4.png 0.985 diff.png");
}

fn main() {
	let args: Vec<String> = env::args().collect();
	if args.len() < 3 {
		usage();
		process::exit(1);
	}

	let image_a_path = &args[1];
	let image_b_path = &args[2];
	let threshold = args
		.get(3)
		.and_then(|s| s.parse::<f64>().ok())
		.unwrap_or(0.985);
	let diff_output = args.get(4);

	let image_a = match image::open(image_a_path) {
		Ok(img) => img,
		Err(err) => {
			eprintln!("Failed to open first image '{}': {}", image_a_path, err);
			process::exit(1);
		}
	};
	let image_b = match image::open(image_b_path) {
		Ok(img) => img,
		Err(err) => {
			eprintln!("Failed to open second image '{}': {}", image_b_path, err);
			process::exit(1);
		}
	};

	// 1) RGB hybrid compare (good default for screenshot-like inputs)
	let rgb_a = image_a.to_rgb8();
	let rgb_b = image_b.to_rgb8();
	let hybrid = match rgb_hybrid_compare(&rgb_a, &rgb_b) {
		Ok(result) => result,
		Err(err) => {
			eprintln!("RGB hybrid comparison failed: {}", err);
			process::exit(1);
		}
	};

	// 2) Grayscale structure compare (MSSIM + RMS)
	let gray_a = image_a.to_luma8();
	let gray_b = image_b.to_luma8();
	let gray_mssim = gray_similarity_structure(&Algorithm::MSSIMSimple, &gray_a, &gray_b);
	let gray_rms = gray_similarity_structure(&Algorithm::RootMeanSquared, &gray_a, &gray_b);

	// 3) Grayscale histogram compare
	let hist_hellinger = gray_similarity_histogram(Metric::Hellinger, &gray_a, &gray_b);

	println!("== image-compare demo ==");
	println!("image_a: {}", image_a_path);
	println!("image_b: {}", image_b_path);
	println!("threshold: {}", threshold);
	println!();
	println!("RGB hybrid score: {:.6}", hybrid.score);
	println!("changed (score < threshold): {}", hybrid.score < threshold);

	match gray_mssim {
		Ok(v) => println!("Gray MSSIM score: {:.6}", v.score),
		Err(e) => println!("Gray MSSIM score: error ({})", e),
	}
	match gray_rms {
		Ok(v) => println!("Gray RMS score: {:.6}", v.score),
		Err(e) => println!("Gray RMS score: error ({})", e),
	}
	match hist_hellinger {
		Ok(v) => println!("Gray histogram (Hellinger): {:.6}", v),
		Err(e) => println!("Gray histogram (Hellinger): error ({})", e),
	}

	if let Some(path) = diff_output {
		let diff_img = hybrid.image.to_color_map();
		if let Err(err) = diff_img.save(path) {
			eprintln!("Failed to save diff image '{}': {}", path, err);
			process::exit(1);
		}
		println!("diff image saved: {}", path);
	}
}
