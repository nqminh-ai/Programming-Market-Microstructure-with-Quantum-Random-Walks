.PHONY: all full data simulate test report verify research clean rebuild_data

# Default target: assumes `data`/`simulate` outputs already exist under
# results/. On a fresh clone (or after changing raw input data), run
# `make full` instead -- `all` alone will fail with missing-input errors
# since it does not (re)run the earlier phony phases.
all: test report

# Full pipeline from raw data through the final report, in dependency
# order. Use this on a fresh clone.
full: data simulate test report

# Phase 2: Data processing and feature engineering
data:
	python -m scripts.pipelines.phase2_pipeline process
	python -m scripts.pipelines.phase2_pipeline features --obi-source trade_imbalance
	python -m scripts.pipelines.phase2_pipeline checkpoint

# Phase 4 & 5: Simulation, benchmarking and statistical testing
simulate:
	python -m scripts.pipelines.phase4_pipeline
	python -m scripts.pipelines.phase5_pipeline

# Run all automated tests
test:
	python -m pytest

# Phase 6: Visualization and reporting
report:
	python -m scripts.pipelines.phase6_pipeline

# Verify the shipped research artifacts are present, provenanced and honestly
# labelled exploratory -- seconds, no reruns. See REPRODUCE.md (level 1).
verify:
	python -m scripts.operations.reproduce
	python -m pytest tests/test_report_numbers.py tests/test_reproduce.py -q

# Print the exact commands that regenerate the §5b-5e research artifacts the
# report's conclusions rest on. These are not run automatically: some take
# hours over hundreds of millions of rows. See REPRODUCE.md (level 3).
research:
	python -m scripts.operations.reproduce --commands

# Clean up results and reports (caution). Requires a POSIX shell (Git Bash,
# WSL); on native Windows PowerShell, run the underlying `python -m
# scripts.pipelines.phase<N>_pipeline` commands directly instead of `make`,
# as already documented in docs/bao_cao_toan_dien.md.
clean:
	rm -rf results/*.csv
	rm -rf figures/*.png figures/*.gif
	rm -rf docs/final_report.pdf docs/presentation_slides.pdf

# Special target to rebuild invalidated historical data (June 1-7)
rebuild_data: data
