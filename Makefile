# Variables
PYTHON=python3
PIP=pip
REQUIREMENTS=requirements.txt
APP=src.main
YOLOV5_DIR=content/yolov5

# Default target
all: clone install run

# Install dependencies
install:
	$(PIP) install -r $(REQUIREMENTS)

clone:
	git clone https://github.com/ultralytics/yolov5.git $(YOLOV5_DIR)

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