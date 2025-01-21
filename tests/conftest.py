import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests that take a long time to run")


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests"
    )


@pytest.fixture(autouse=True)
def skip_by_default(request):
    if request.node.get_closest_marker("slow"):
        if not request.config.getoption("--run-slow"):
            pytest.skip("Only run with --run-slow flag")
