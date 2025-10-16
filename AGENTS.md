# Agent Guidelines for YOLO Training Template

## Commands
- **Install**: `pip install -r requirements.txt`
- **Train**: `python scripts/main.py --dataset <kaggle-handle> --nc <num-classes> --names <class-names>`
- **Inference**: `python scripts/inference.py --model <model-path> --input <image/video/webcam>`
- **Lint**: No linter configured - use `python -m py_compile` for syntax check
- **Test**: No test framework configured - manually test scripts
- **Format**: No formatter configured - follow PEP 8 manually

## Code Style
- **Imports**: Standard library first, then third-party, then local. One per line.
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- **Types**: No type hints required but preferred for clarity
- **Formatting**: Lines <80 chars, 4-space indentation, PEP 8 compliant
- **Docstrings**: Required for all functions using triple quotes
- **Error Handling**: Use try/except with specific exceptions, log errors with logging module
- **Logging**: Use logging module with INFO level, format: '%(asctime)s - %(levelname)s - %(message)s'
- **Arguments**: Use argparse for CLI scripts, required args first, defaults for optional