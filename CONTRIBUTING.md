# Contributing to LLM Hook Analysis Framework

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🤝 Ways to Contribute

### 1. Report Bugs

Found a bug? Please create an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, PyTorch version, OS)
- Error messages and stack traces

### 2. Suggest Features

Have an idea for a new feature?
- Check existing issues first
- Create a new issue with the "enhancement" label
- Describe the use case and benefits
- Provide examples if possible

### 3. Improve Documentation

Documentation improvements are always welcome:
- Fix typos and grammar
- Add examples
- Clarify existing documentation
- Translate documentation
- Write tutorials

### 4. Submit Code

Want to contribute code? Great!
- Start with good first issues
- Fork the repository
- Create a feature branch
- Make your changes
- Submit a pull request

---

## 🚀 Getting Started

### Setup Development Environment

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Hook_Technique.git
   cd Hook_Technique
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks** (optional but recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

### Project Structure

```
Hook_Technique/
├── llm_hooks/          # Main package
│   ├── core/           # Core functionality
│   ├── pytorch/        # PyTorch hooks
│   ├── attention/      # Attention monitoring
│   ├── activations/    # Activation analysis
│   ├── fisher/         # Fisher information
│   ├── gradients/      # Gradient tracking
│   ├── cuda/           # CUDA profiling
│   ├── tensorrt/       # TensorRT support
│   ├── visualization/  # Visualization tools
│   ├── analysis/       # Analysis tools
│   └── utils/          # Utilities
├── examples/           # Example scripts
├── tutorials/          # Tutorial scripts
├── tests/              # Test suite
├── docs/               # Documentation
└── scripts/            # Utility scripts
```

---

## 📝 Coding Guidelines

### Code Style

We follow PEP 8 with some modifications:
- Maximum line length: 100 characters
- Use type hints where possible
- Use docstrings for all public functions and classes

**Format your code**:
```bash
# Using black
black llm_hooks/

# Using flake8 for linting
flake8 llm_hooks/
```

### Docstring Format

Use Google-style docstrings:

```python
def my_function(arg1: str, arg2: int) -> bool:
    """
    Short description of the function.

    Longer description if needed, explaining behavior,
    edge cases, etc.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When invalid input

    Example:
        >>> my_function("test", 42)
        True
    """
    pass
```

### Type Hints

Use type hints for function signatures:

```python
from typing import List, Dict, Optional, Any

def process_data(
    data: List[Dict[str, Any]],
    threshold: float = 0.5,
    verbose: bool = False
) -> Optional[Dict[str, float]]:
    """Process data with given threshold."""
    pass
```

---

## 🧪 Testing

### Writing Tests

- Write tests for all new features
- Place tests in `tests/` directory
- Use pytest framework
- Aim for >80% code coverage

**Test file naming**: `test_<module_name>.py`

**Example test**:
```python
import pytest
from llm_hooks import HookManager


def test_hook_manager_creation():
    """Test HookManager initialization."""
    manager = HookManager(name="test")
    assert manager.name == "test"
    assert len(manager.hooks) == 0


def test_hook_registration():
    """Test hook registration."""
    from llm_hooks.pytorch import ForwardHook

    manager = HookManager()
    hook = ForwardHook(layer_name="test")
    manager.register(hook)

    assert len(manager.hooks) == 1
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_basic.py

# Run with coverage
pytest --cov=llm_hooks --cov-report=html

# Run with verbose output
pytest -v
```

---

## 📦 Adding New Hooks

Creating a new hook type:

### 1. Create Hook Class

```python
# llm_hooks/my_module/my_hook.py

from llm_hooks.core.base_hook import BaseHook
from llm_hooks.core.hook_registry import register_hook


@register_hook("my_hook")  # Register with name
class MyHook(BaseHook):
    """Description of what your hook does."""

    def __init__(self, param1: str, param2: int = 10, name: str = None):
        """
        Initialize the hook.

        Args:
            param1: Description
            param2: Description
            name: Optional hook name
        """
        super().__init__(name=name or "MyHook")
        self.param1 = param1
        self.param2 = param2

    def setup(self, model):
        """Setup hook on model."""
        # Your setup logic
        for name, module in model.named_modules():
            if self._should_hook(module):
                handle = module.register_forward_hook(self._hook_fn)
                self._hook_handles.append(handle)

    def teardown(self):
        """Remove hooks and cleanup."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def _hook_fn(self, module, input, output):
        """Hook function."""
        if not self.enabled:
            return

        # Capture and analyze
        data = self._analyze(output)

        # Create result
        result = self.create_result(
            hook_type='my_hook',
            data=data,
            layer_name=self._get_layer_name(module)
        )
        self.add_result(result)

    def _analyze(self, tensor):
        """Analyze captured data."""
        # Your analysis logic
        return {'my_metric': tensor.mean().item()}
```

### 2. Add __init__.py

```python
# llm_hooks/my_module/__init__.py

from llm_hooks.my_module.my_hook import MyHook

__all__ = ["MyHook"]
```

### 3. Write Tests

```python
# tests/test_my_hook.py

import pytest
import torch
from llm_hooks import HookManager
from llm_hooks.my_module import MyHook


def test_my_hook():
    """Test MyHook functionality."""
    # Your test implementation
    pass
```

### 4. Add Documentation

- Update `docs/api.md` with new hook documentation
- Add example in `examples/`
- Update README if necessary

---

## 📚 Documentation

### Building Documentation

We use Markdown for documentation.

**Documentation structure**:
- `docs/` - Main documentation
- Inline code comments
- Docstrings in code
- Examples in `examples/`
- Tutorials in `tutorials/`

### Writing Documentation

Good documentation should:
- Be clear and concise
- Include examples
- Explain why, not just what
- Cover edge cases
- Link to related topics

---

## 🔄 Pull Request Process

### Before Submitting

1. **Update your fork**:
   ```bash
   git checkout main
   git pull upstream main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/my-feature
   ```

3. **Make your changes**:
   - Write code
   - Add tests
   - Update documentation

4. **Run tests**:
   ```bash
   pytest
   ```

5. **Format code**:
   ```bash
   black llm_hooks/
   flake8 llm_hooks/
   ```

### Submitting PR

1. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

2. **Push to your fork**:
   ```bash
   git push origin feature/my-feature
   ```

3. **Create Pull Request**:
   - Go to GitHub
   - Click "New Pull Request"
   - Select your branch
   - Fill out the template

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe how you tested your changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
```

### Review Process

- Maintainers will review your PR
- Address feedback if requested
- Once approved, PR will be merged
- Your contribution will be acknowledged

---

## 🐛 Debugging Tips

### Common Issues

1. **Import errors**:
   ```bash
   pip install -e .  # Reinstall in dev mode
   ```

2. **Test failures**:
   ```bash
   pytest -v  # Verbose output
   pytest --pdb  # Drop into debugger on failure
   ```

3. **Hook not capturing data**:
   - Check layer names
   - Ensure hooks applied before inference
   - Verify model in correct mode (train/eval)

---

## 💬 Communication

### Questions?

- Check [FAQ](docs/faq.md)
- Search existing [Issues](https://github.com/GeoffreyWang1117/Hook_Technique/issues)
- Start a [Discussion](https://github.com/GeoffreyWang1117/Hook_Technique/discussions)

### Reporting Security Issues

Do not create public issues for security vulnerabilities.
Email the maintainers directly.

---

## 📜 Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone.

### Our Standards

**Positive behavior**:
- Using welcoming and inclusive language
- Respecting differing viewpoints
- Accepting constructive criticism
- Focusing on what is best for the community

**Unacceptable behavior**:
- Trolling, insulting, or derogatory comments
- Personal or political attacks
- Public or private harassment
- Publishing others' private information

### Enforcement

Violations may result in:
- Warning
- Temporary ban
- Permanent ban

Report violations to project maintainers.

---

## 🎉 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Credited in release notes
- Acknowledged in documentation

Thank you for contributing! 🙏

---

## 📝 License

By contributing, you agree that your contributions will be licensed under the project's MIT License.
