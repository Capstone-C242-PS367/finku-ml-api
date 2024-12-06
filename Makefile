# Variables
PYTHON=python3
PIP=pip
REQUIREMENTS=requirements.txt
APP=src.main

# Default target
all: install run

# Install dependencies
install:
	$(PIP) install -r $(REQUIREMENTS)

# Run the application
run:
	$(PYTHON) -m $(APP)

# Clean up (optional)
clean:
	rm -rf __pycache__  # Remove Python cache files
	rm -rf uploads/*    # Clear the uploads directory (if needed)

# Help target
help:
	@echo "Makefile commands:"
	@echo "  make install   - Install dependencies"
	@echo "  make run       - Run the Flask application"
	@echo "  make clean     - Clean up cache and uploads"
	@echo "  make help      - Show this help message"