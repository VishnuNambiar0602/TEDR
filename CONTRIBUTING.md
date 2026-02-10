# Contributing to TEDR

Thank you for your interest in contributing to TEDR! This document provides guidelines and instructions for contributing.

## 🎯 Ways to Contribute

1. **Report Bugs**: Found a bug? Open an issue with detailed information
2. **Suggest Features**: Have an idea? Share it in the issues
3. **Improve Documentation**: Fix typos, add examples, clarify instructions
4. **Submit Code**: Fix bugs, add features, improve performance
5. **Share Datasets**: Contribute Indian road datasets for training

## 🚀 Getting Started

### Setting Up Development Environment

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/TEDR.git
   cd TEDR
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```
5. Create a branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 📝 Coding Standards

### Python Style Guide

- Follow PEP 8 style guide
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise
- Use type hints where appropriate

Example:
```python
def detect_objects(image: Image.Image, threshold: float = 0.7) -> Dict[str, Any]:
    """
    Detect objects in an image.
    
    Args:
        image: PIL Image object
        threshold: Confidence threshold for detections
        
    Returns:
        Dictionary containing detection results
    """
    # Implementation here
    pass
```

### JavaScript Style Guide

- Use modern ES6+ syntax
- Use meaningful variable names
- Add comments for complex logic
- Keep functions small and focused
- Use async/await for asynchronous operations

### Commit Messages

Follow conventional commit format:
- `feat: Add new feature`
- `fix: Fix bug in detection`
- `docs: Update README`
- `style: Format code`
- `refactor: Refactor model loading`
- `test: Add API tests`
- `chore: Update dependencies`

## 🧪 Testing

### Running Tests

```bash
# Test system components
python test_system.py

# Test API (requires server running)
python test_api.py --image path/to/test/image.jpg
```

### Adding Tests

When adding new features, include tests:
- Unit tests for new functions
- Integration tests for API endpoints
- Example usage in documentation

## 📖 Documentation

### Code Documentation

- Add docstrings to all public functions and classes
- Include parameter descriptions and return types
- Add usage examples where helpful

### User Documentation

When adding features, update:
- README.md with feature description
- API documentation if adding endpoints
- Configuration guide if adding settings
- QUICKSTART.md if changing installation

## 🎨 Feature Requests

### Proposing New Features

1. Check existing issues to avoid duplicates
2. Open an issue with:
   - Clear description of the feature
   - Use case and motivation
   - Proposed implementation (optional)
   - Examples or mockups (if applicable)

### Priority Features

Current priorities:
- Auto rickshaw detection improvements
- Video detection support
- Mobile app integration
- Performance optimization
- Dataset collection tools

## 🐛 Bug Reports

### Creating Bug Reports

Include:
1. Description of the bug
2. Steps to reproduce
3. Expected behavior
4. Actual behavior
5. Environment details:
   - Python version
   - OS and version
   - GPU/CPU setup
   - Dependencies versions

Example:
```
Bug: Detection fails on large images

Steps to reproduce:
1. Upload image larger than 4000x3000
2. Click detect
3. Error occurs

Expected: Image should be processed
Actual: Server returns 500 error

Environment:
- Python 3.10
- Ubuntu 22.04
- CUDA 11.8
- 8GB RAM
```

## 🔄 Pull Request Process

1. **Create Issue First**: For major changes, create an issue first to discuss
2. **Branch Naming**: Use descriptive names like `feature/auto-rickshaw-training` or `fix/memory-leak`
3. **Code Quality**: Ensure code follows style guidelines
4. **Tests**: Add tests for new features
5. **Documentation**: Update relevant documentation
6. **Commits**: Use clear, conventional commit messages
7. **Pull Request**:
   - Write clear PR description
   - Link related issues
   - Add screenshots for UI changes
   - Wait for review

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Commits are clear and atomic
- [ ] Branch is up to date with main
- [ ] No merge conflicts

## 🌟 Areas for Contribution

### Beginner-Friendly

- Fix typos in documentation
- Add code comments
- Improve error messages
- Add usage examples
- Test on different platforms

### Intermediate

- Improve UI/UX
- Add API endpoints
- Optimize preprocessing
- Add visualization options
- Improve configuration system

### Advanced

- Model optimization
- Custom training pipelines
- Multi-GPU support
- Model quantization
- Mobile deployment
- Video processing

## 📊 Dataset Contributions

### Sharing Datasets

If you have Indian road datasets:
1. Ensure proper licensing and permissions
2. Format as COCO annotations
3. Include variety of scenarios
4. Document collection conditions
5. Share via issue or external link

### Dataset Requirements

- Proper licensing (CC-BY, CC0, or similar)
- COCO format annotations
- High quality images
- Diverse scenarios
- Privacy considerations met

## 💬 Community Guidelines

- Be respectful and inclusive
- Help others learn
- Share knowledge and resources
- Give constructive feedback
- Assume positive intent

## 📧 Questions?

- Open an issue for technical questions
- Check existing documentation first
- Provide context and examples
- Be patient waiting for responses

## 🙏 Recognition

Contributors will be:
- Listed in README.md
- Credited in release notes
- Appreciated in the community

---

Thank you for contributing to TEDR! Together we can make Indian roads safer with AI. 🚗🛺🚙
