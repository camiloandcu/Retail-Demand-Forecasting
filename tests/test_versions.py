"""Environment metadata tests."""

from retail_forecast.versions import capture_versions


def test_versions_include_core_runtime() -> None:
    versions = capture_versions()
    assert versions["python"]
    assert versions["packages"]["retail-forecast"] == "0.1.0"
    assert versions["packages"]["numpy"]
