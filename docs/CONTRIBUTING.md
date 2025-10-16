# Contributing to YOLO Training Template

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## How to Contribute

1. **Fork the Repository**: Create a fork of this repo on GitHub.
2. **Clone Your Fork**: `git clone https://github.com/your-username/yolo-training-template.git`
3. **Create a Branch**: `git checkout -b feature/your-feature-name`
4. **Make Changes**: Implement your changes.
5. **Test**: Ensure your changes work as expected.
6. **Commit**: `git commit -m "Add your message"`
7. **Push**: `git push origin feature/your-feature-name`
8. **Create a Pull Request**: Submit a PR with a clear description.

## Code Style

- Follow PEP 8 for Python code.
- Use meaningful variable and function names.
- Add docstrings to functions.
- Keep lines under 80 characters.

## Reporting Issues

- Use GitHub Issues to report bugs or request features.
- Provide detailed steps to reproduce bugs.
- Include environment details (Python version, OS, etc.).

## Running on Google Colab

To run the Jupyter notebook on Google Colab:

1. **Open Colab**: Go to [Google Colab](https://colab.research.google.com/).
2. **Upload Notebook**: Upload `yolo_template.ipynb` to Colab.
3. **Install Dependencies**: Run the first cell in the notebook to install requirements.
4. **Run Cells**: Execute the cells sequentially for training and inference.
5. **GPU Support**: Enable GPU in Colab (Runtime > Change runtime type > Hardware accelerator > GPU).
6. **Data Access**: For Kaggle datasets, upload your `kaggle.json` or use Colab's Kaggle integration.
7. **Save Outputs**: Use Colab's file system to save models and results.

Note: Colab has resource limits; for long training, consider local setup.