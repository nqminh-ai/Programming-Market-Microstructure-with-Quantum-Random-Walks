.PHONY: all data simulate test report clean rebuild_data

# Default target
all: test report

# Phase 2: Data processing and feature engineering
data:
	python scripts/phase2_pipeline.py process
	python scripts/phase2_pipeline.py features --obi-source trade_imbalance
	python scripts/phase2_pipeline.py checkpoint

# Phase 4 & 5: Simulation, benchmarking and statistical testing
simulate:
	python scripts/phase4_pipeline.py
	python scripts/phase5_pipeline.py

# Run all automated tests
test:
	python -m pytest

# Phase 6: Visualization and reporting
report:
	python scripts/phase6_pipeline.py

# Clean up results and reports (caution)
clean:
	rm -rf results/*.csv
	rm -rf figures/*.png figures/*.gif
	rm -rf docs/final_report.pdf docs/presentation_slides.pdf

# Special target to rebuild invalidated historical data (June 1-7)
rebuild_data: data
