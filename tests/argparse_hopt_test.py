import pytest


def test_hello():
    print("hello travis")


if __name__ == '__main__':
    pytest.main([__file__])
