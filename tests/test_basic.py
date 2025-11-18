"""
Basic tests for core functionality.
"""

import pytest
import torch
import torch.nn as nn
from llm_hooks import HookManager
from llm_hooks.pytorch import ForwardHook, BackwardHook


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def test_hook_manager_creation():
    """Test HookManager creation."""
    manager = HookManager(name="test")
    assert manager.name == "test"
    assert len(manager.hooks) == 0
    assert not manager._is_setup


def test_forward_hook_registration():
    """Test forward hook registration."""
    manager = HookManager()
    hook = ForwardHook(layer_name="fc1")
    manager.register(hook)

    assert len(manager.hooks) == 1
    assert hook in manager.hooks


def test_forward_hook_capture():
    """Test forward hook data capture."""
    model = SimpleModel()
    manager = HookManager()

    hook = ForwardHook(layer_name="fc1", compute_stats=True)
    manager.register(hook)
    manager.apply_to_model(model)

    # Run inference
    x = torch.randn(2, 10)
    with torch.no_grad():
        output = model(x)

    # Check results
    results = manager.get_results()
    assert len(results) == 1

    result = results.results[0]
    assert result.layer_name == "fc1"
    assert result.hook_type == "forward"
    assert 'output' in result.data

    manager.teardown()


def test_backward_hook_capture():
    """Test backward hook data capture."""
    model = SimpleModel()
    model.train()

    manager = HookManager()
    hook = BackwardHook(layer_name="fc1", compute_stats=True)
    manager.register(hook)
    manager.apply_to_model(model)

    # Run forward and backward
    x = torch.randn(2, 10)
    target = torch.randint(0, 5, (2,))

    output = model(x)
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()

    # Check results
    results = manager.get_results()
    assert len(results) >= 1  # At least one backward result

    manager.teardown()


def test_multiple_hooks():
    """Test multiple hooks on same model."""
    model = SimpleModel()
    manager = HookManager()

    forward_hook = ForwardHook(layer_name="fc1")
    backward_hook = BackwardHook(layer_name="fc1")

    manager.register(forward_hook)
    manager.register(backward_hook)
    manager.apply_to_model(model)

    assert len(manager.hooks) == 2

    manager.teardown()


def test_hook_enable_disable():
    """Test enabling/disabling hooks."""
    hook = ForwardHook(layer_name="fc1")

    assert hook.enabled

    hook.disable()
    assert not hook.enabled

    hook.enable()
    assert hook.enabled


def test_clear_results():
    """Test clearing results."""
    model = SimpleModel()
    manager = HookManager()

    hook = ForwardHook(layer_name="fc1")
    manager.register(hook)
    manager.apply_to_model(model)

    # Generate results
    x = torch.randn(2, 10)
    with torch.no_grad():
        _ = model(x)

    assert len(manager.get_results()) > 0

    # Clear results
    manager.clear_results()
    assert len(manager.get_results()) == 0

    manager.teardown()


def test_context_manager():
    """Test using HookManager as context manager."""
    model = SimpleModel()

    with HookManager() as manager:
        hook = ForwardHook(layer_name="fc1")
        manager.register(hook)
        manager.apply_to_model(model)

        x = torch.randn(2, 10)
        with torch.no_grad():
            _ = model(x)

    # Hooks should be cleaned up after context
    assert not manager._is_setup


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
