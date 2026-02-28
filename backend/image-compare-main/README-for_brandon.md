# README for Brandon: RGB Hybrid Score Only

This doc is intentionally minimal.  
Goal: run one command that returns a single number (RGB hybrid similarity score) so it can be integrated into your framework.
## Get Started

### 1) Install Rust (Cargo)

On Windows (PowerShell):

```powershell
winget install Rustlang.Rustup
```

If `cargo` is still not recognized, initialize the toolchain directly:

```powershell
& "$env:USERPROFILE\.cargo\bin\rustup.exe" default stable
```

Open a new terminal, then verify:

```powershell
cargo --version
rustc --version
```

On macOS (Terminal):

```bash
brew install rustup-init
rustup-init -y
source "$HOME/.cargo/env"
```

Then verify:

```bash
cargo --version
rustc --version
```

## What to Run

From `image-compare-main`:

```powershell
cargo run --example rgb_hybrid_score -- <image_a> <image_b>
```

macOS command is the same:

```bash
cargo run --example rgb_hybrid_score -- <image_a> <image_b>
```

Example:

```powershell
cargo run --example rgb_hybrid_score -- ../scripts/image_6.png ../scripts/image_7.png
```

## Output

The command prints only one line:

```text
0.716054
```

- `1.0` = images are effectively identical
- lower = more visual change

## Script Location

- `examples/rgb_hybrid_score.rs`

It uses:

- `image_compare::rgb_hybrid_compare`
- `image::open(...).into_rgb8()`

## Integration Contract

Treat this as a CLI function:

**Input**
- path to image A
- path to image B

**Output**
- stdout: decimal score (string), parse as `float`
- non-zero exit code on failure

## Quick Node Integration Pattern

Use child process and parse stdout:

```js
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

async function getRgbHybridScore(imageAPath, imageBPath) {
  const { stdout } = await execFileAsync(
    "cargo",
    ["run", "--example", "rgb_hybrid_score", "--", imageAPath, imageBPath],
    { cwd: "image-compare-main" }
  );

  const score = Number(stdout.trim());
  if (!Number.isFinite(score)) throw new Error(`Invalid score output: ${stdout}`);
  return score;
}
```

## Suggested Threshold

Start with:

- `changed = score < 0.7`

Then tune based on your real screenshot stream.
