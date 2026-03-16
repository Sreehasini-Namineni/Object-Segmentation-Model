# CS3630 Lab 1: RGBD Tabletop Instance Segmentation

## Getting Started

- This lab introduces tabletop instance segmentation on the OCID dataset.
- Install dependencies and verify that you can run the starter code.
- Explore creative segmentation strategies beyond the baseline RANSAC.

## Logistics

- Implement RANSAC baseline via `ransac_plane.py` to segment objects on a tabletop.
- Implement your own custom method to outperform baseline.
- Submit via Gradescope.

## Rubric
- +10 pts: Correct `fit_plane_from_3pts()` implementation
- +10 pts: Correct `ransac_plane()` implementation
- +10 pts: RANSAC algorithm correctness (integration test)
- +30 pts: RANSAC baseline passes easy dataset performance threshold
- +10 pts: Custom method implemented and passes easy dataset
- +30 pts: Custom method outperforms baseline on hard dataset
- Total: 100 pts

## Installation

### Prerequisites

- Python 3.8+ and conda (Anaconda or Miniconda)
- Download Miniconda from: https://docs.conda.io/en/latest/miniconda.html

### Setup Commands

```bash
# Create conda environment
conda create -n cs3630lab1 python=3.10
conda activate cs3630lab1

# Install dependencies
conda install numpy scipy matplotlib

# Verify installation
python -c "import numpy; print(f'NumPy {numpy.__version__} installed')"

# Test your setup
python3 test.py --dataset dataset/easy --config config.json
```

If you see metrics output, your environment is ready!

### Visualization
If you want to see your segmentation output to understand where your algorithm fails and iterate on improvements, try:

```bash
python3 test.py --dataset dataset/hard --visualize --vis-dir visualizations/
```

## Project Structure

```console
├── README.md
├── baseline.py              # Baseline RANSAC
├── solution.py              # TODO: call your custom method here
├── test.py                  # Testing script
├── dataset/
│   ├── easy/                # Testing dataset
├── utils/
│   ├── dataset_loader.py
│   ├── evaluation.py
│   ├── ocid_io.py
│   ├── ransac_plane.py      # TODO: Implement RANSAC here!
│   ├── segmentation.py
│   ├── segmentation_types.py
│   └── visualize.py
└── .gitignore
```

## Tips
- Start by implementing RANSAC in `utils/ransac_plane.py` following the TODOs
- Use `baseline.py` as a reference for the full pipeline
- Test frequently with `test.py`. To understand what we score you on, read the instruction pdf file.
- For custom method, consider:
  - Using RGB color information
  - Depth-based clustering
  - Region growing algorithms
  - Combining multiple approaches
- If PQ is high but results look wrong, check FP/FN: PQ penalizes missing/extra instances via RQ
- Do **not** hard-code paths; use relative paths
- Keep function signatures unchanged
- Go to office hours or post questions if you are unsure about anything

## Submission

Run this command from the lab directory to create your submission zip for Gradescope:

**Linux/Mac:**
```bash
zip -r lab1submission.zip utils/<your_custom_files>.py utils/ransac_plane.py solution.py  <your_custom_files>.py
```

**Windows:**
You can either zip it in your own way or use Powershell:
```powershell
Compress-Archive -Path utils/<your_custom_files>.py,utils/ransac_plane.py,solution.py,<your_custom_files>.py -DestinationPath lab1submission.zip -Force
```
Make sure that the zip files contains all the custom files you wrote and remember to replace baseline_solve with your own method in `solution.py` to get full score!
