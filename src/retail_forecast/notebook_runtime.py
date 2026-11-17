"""Pinned source setup with locked local and non-disruptive Colab installation."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import re
import shutil
import site
import subprocess
import sys
import tempfile
from pathlib import Path

SUPPORTED_PYTHON = ((3, 10), (3, 11), (3, 12), (3, 13))


def check_python(version: tuple[int, int] | None = None) -> None:
    """Reject unsupported interpreters before installation or data acquisition."""
    if (version or sys.version_info[:2]) not in SUPPORTED_PYTHON:
        raise RuntimeError("Use Python 3.10–3.13; this runtime is unsupported. Restart with it.")


def git(root: Path, *args: str) -> str:
    """Read Git identity without changing contributor branches."""
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def validate_checkout(root: Path, revision: str, origin: str) -> Path:
    """Refuse incomplete, mismatched or dirty repositories without mutation."""
    check_python()
    root = root.resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("CODE_REVISION must be a reviewed 40-character commit SHA.")
    try:
        if git(root, "rev-parse", "--show-toplevel") != str(root):
            raise RuntimeError("Select the checkout root, not a nested directory.")
        if git(root, "remote", "get-url", "origin") != origin:
            raise RuntimeError("Checkout origin mismatch; use a new directory.")
        if git(root, "rev-parse", "HEAD") != revision:
            raise RuntimeError("Checkout revision mismatch; use the pinned checkout separately.")
        if git(root, "status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError("Checkout has user changes; preserve it and use a new directory.")
    except subprocess.CalledProcessError:
        raise RuntimeError("Incomplete checkout; preserve it and use a new directory.") from None
    for filename in ("pyproject.toml", "uv.lock", "src/retail_forecast/notebook_runtime.py"):
        if not (root / filename).is_file():
            raise RuntimeError("Incomplete checkout; use a new directory.")
    return root


def verify_loaded_source(root: Path) -> None:
    """An already imported foreign package cannot be repaired by changing paths."""
    for name, module in tuple(sys.modules.items()):
        if name == "retail_forecast" or name.startswith("retail_forecast."):
            source = getattr(module, "__file__", None)
            if source and not Path(source).resolve().is_relative_to(root / "src"):
                raise RuntimeError(
                    "Package already imported from another checkout. Restart kernel."
                )


def clear_loaded_bytecode(package: str) -> None:
    """Remove cached code for one package when its files were just replaced."""
    for name, module in tuple(sys.modules.items()):
        if name != package and not name.startswith(f"{package}."):
            continue
        cached = getattr(module, "__cached__", None)
        if cached:
            try:
                Path(cached).unlink(missing_ok=True)
            except OSError:
                raise RuntimeError(
                    f"Could not clear stale {package} bytecode. Restart runtime manually."
                ) from None


def selected_profile(
    root: Path, *, models: bool = False, colab: bool = False, neural: bool = False
) -> Path:
    """Select the locked profile without Colab's managed notebook packages."""
    if neural:
        return root / "constraints" / ("06-colab-neural.txt" if colab else "05-notebook-neural.txt")
    if colab:
        filename = "04-colab-models.txt" if models else "03-colab.txt"
    else:
        filename = "02-notebook-models.txt" if models else "01-notebook.txt"
    return root / "constraints" / filename


def profile_name(*, models: bool, colab: bool, neural: bool = False) -> str:
    if neural:
        return "colab-neural" if colab else "neural"
    if colab:
        return "colab-models" if models else "colab"
    return "models" if models else "notebook"


def profile_distributions(profile: Path) -> set[str]:
    """Read exported exact pins so kernel checks cover only installed packages."""
    return {
        re.sub(r"[-_.]+", "-", match.group(1)).lower()
        for line in profile.read_text().splitlines()
        if (match := re.match(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==", line))
    }


def install(root: Path, *, models: bool = False, colab: bool = False, neural: bool = False) -> None:
    """Install hashed dependencies into this interpreter, then the pinned project."""
    profile = selected_profile(root, models=models, colab=colab, neural=neural)
    lock_hash = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    if f"# uv.lock SHA256: {lock_hash}" not in profile.read_text():
        raise RuntimeError("Constraints do not match uv.lock; regenerate before publishing.")
    profile_hash = hashlib.sha256(profile.read_bytes()).hexdigest()
    name = profile_name(models=models, colab=colab, neural=neural)
    marker = (
        Path(tempfile.gettempdir())
        / f"retail-forecast-{name}-py{sys.version_info.major}{sys.version_info.minor}.json"
    )
    if importlib.util.find_spec("pip") is not None:
        command = [sys.executable, "-m", "pip", "install"]
        freeze_command = [sys.executable, "-m", "pip", "freeze"]
    elif shutil.which("uv"):
        command = ["uv", "pip", "install", "--python", sys.executable]
        freeze_command = ["uv", "pip", "freeze", "--python", sys.executable]
    else:
        raise RuntimeError("Install pip or uv in this Python runtime, then retry.")

    def freeze_hash() -> str:
        result = subprocess.run(freeze_command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError("Could not verify installed environment; retry setup.")
        return hashlib.sha256(result.stdout.encode()).hexdigest()

    if marker.is_file():
        try:
            receipt = json.loads(marker.read_text())
        except (OSError, json.JSONDecodeError):
            receipt = {}
        if (
            receipt.get("project_root") == str(root.resolve())
            and receipt.get("lock_sha256") == lock_hash
            and receipt.get("profile_sha256") == profile_hash
            and receipt.get("python") == list(sys.version_info[:2])
            and receipt.get("freeze_sha256") == freeze_hash()
        ):
            return

    # Output is bounded and never includes environment variables or credentials.
    for arguments in (
        ["--require-hashes", "-r", str(profile)],
        ["--no-deps", "--no-build-isolation", "-e", str(root)],
    ):
        result = subprocess.run(command + arguments, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(
                "Locked installation failed. Check connectivity/runtime; retry setup."
            )
    receipt = {
        "project_root": str(root.resolve()),
        "lock_sha256": lock_hash,
        "profile_sha256": profile_hash,
        "python": list(sys.version_info[:2]),
        "freeze_sha256": freeze_hash(),
    }
    marker.write_text(json.dumps(receipt))
    # Process newly installed editable .pth files in the already running kernel.
    for directory in site.getsitepackages():
        site.addsitedir(directory)
    importlib.invalidate_caches()


def installed_versions() -> dict[str, str]:
    """Snapshot active distribution versions without importing their modules."""
    names = {
        re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    return {
        name: importlib.metadata.version(name)
        for name in sorted(names)
        if name != "retail-forecast"
    }


def pip_diagnostic(result: subprocess.CompletedProcess) -> str:
    """Bound resolver diagnostics and redact URLs and credential assignments."""
    lines = (result.stdout + "\n" + result.stderr).splitlines()
    detail = "\n".join(lines[-14:])
    detail = re.sub(r"https?://\S+", "[URL omitted]", detail)
    detail = re.sub(
        r"(?i)\b(token|password|api[_-]?key|authorization)(\s*[:=]\s*)\S+",
        r"\1\2[omitted]",
        detail,
    )
    return detail[-2400:]


def install_colab(root: Path, *, models: bool = False, neural: bool = False) -> dict:
    """Resolve project requirements while protecting every existing distribution.

    Colab owns its running kernel. The universal lock is used locally, not to
    replace Colab's resident NumPy, plotting, CUDA or notebook dependencies.
    A constrained dry run must succeed before any installation takes place.
    """
    if importlib.util.find_spec("pip") is None:
        raise RuntimeError("Colab setup requires its bundled pip installation.")
    before = installed_versions()
    extras = ["colab"]
    if models:
        extras.append("colab-models")
    if neural:
        extras.append("neural")
    command = [sys.executable, "-m", "pip", "install", "--no-build-isolation"]
    with tempfile.TemporaryDirectory(prefix="retail-colab-setup-") as temporary:
        folder = Path(temporary)
        constraints = folder / "resident.txt"
        constraints.write_text("".join(f"{name}=={v}\n" for name, v in before.items()))
        report_path = folder / "plan.json"
        arguments = ["--constraint", str(constraints), "-e", f"{root}[{','.join(extras)}]"]
        plan = subprocess.run(
            command + ["--dry-run", "--report", str(report_path)] + arguments,
            capture_output=True,
            text=True,
        )
        if plan.returncode:
            raise RuntimeError(
                "Colab dependencies are incompatible with this runtime. "
                "No packages were replaced. Resolver diagnostics:\n" + pip_diagnostic(plan)
            )
        planned = json.loads(report_path.read_text())
        replacements = sorted(
            item["metadata"]["name"]
            for item in planned.get("install", [])
            if re.sub(r"[-_.]+", "-", item["metadata"]["name"]).lower() in before
        )
        if replacements:
            raise RuntimeError(f"Colab setup refused to replace resident packages: {replacements}")
        result = subprocess.run(command + arguments, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(
                "Colab dependency installation failed. Diagnostics:\n" + pip_diagnostic(result)
            )
    for directory in site.getsitepackages():
        site.addsitedir(directory)
    importlib.invalidate_caches()
    after = installed_versions()
    changed = sorted(name for name, version in before.items() if after.get(name) != version)
    if changed:
        raise RuntimeError(f"Colab resident packages unexpectedly changed: {changed}")
    return {
        "policy": "colab-compatible-preserve-installed",
        "extras": extras,
        "preserved_distributions": len(before),
        "added_distributions": sorted(set(after) - set(before)),
        "packages": after,
    }


def bootstrap(
    root: Path,
    revision: str,
    origin: str,
    *,
    models: bool = False,
    colab: bool = False,
    neural: bool = False,
) -> dict:
    """Validate, install and report identities; refuse mixed-source kernel state."""
    root = validate_checkout(root, revision, origin)
    verify_loaded_source(root)
    selected = selected_profile(root, models=models, colab=colab, neural=neural)
    installable = profile_distributions(selected)
    # Dependencies imported before an upgrade remain resident. Detect disagreement
    # after installation and require restart instead of running mixed versions.
    loaded = {}
    stale_modules = set()
    distribution_map = importlib.metadata.packages_distributions()
    for name, module in tuple(sys.modules.items()):
        if "." in name:
            continue
        for distribution in distribution_map.get(name, []):
            if re.sub(r"[-_.]+", "-", distribution).lower() not in installable:
                continue
            version = importlib.metadata.version(distribution)
            resident = getattr(module, "__version__", None)
            # Numeric module versions can retain zero padding removed from wheel
            # metadata (certifi 2026.07.22 == 2026.7.22). Compare release integers;
            # non-release module strings have no general distribution contract.
            if (
                isinstance(resident, str)
                and re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", resident)
                and re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", version)
            ):
                resident_parts = [int(part) for part in resident.split(".")]
                version_parts = [int(part) for part in version.split(".")]
                while resident_parts and resident_parts[-1] == 0:
                    resident_parts.pop()
                while version_parts and version_parts[-1] == 0:
                    version_parts.pop()
                if resident_parts != version_parts:
                    stale_modules.add(name)
            loaded[distribution] = (version, name)
    if stale_modules:
        # Colab can retain timestamp-valid bytecode after pip replaced source
        # files. Clear every affected package before requesting one restart.
        for name in stale_modules:
            clear_loaded_bytecode(name)
        raise RuntimeError("Loaded dependency versions differ. Restart kernel.")
    colab_installation = None
    if colab:
        colab_installation = install_colab(root, models=models, neural=neural)
    elif neural:
        install(root, models=models, neural=True)
    else:
        install(root, models=models)
    changed_modules = set()
    for distribution, (old, name) in loaded.items():
        if old != importlib.metadata.version(distribution):
            changed_modules.add(name)
    if changed_modules:
        for name in changed_modules:
            clear_loaded_bytecode(name)
        raise RuntimeError("Locked dependencies changed during install. Restart kernel.")
    package = importlib.import_module("retail_forecast")
    source = Path(package.__file__).resolve()
    if not source.is_relative_to(root / "src"):
        raise RuntimeError("Installed package resolves another checkout. Restart kernel.")
    distribution = importlib.metadata.distribution("retail-forecast")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct.get("url") != root.as_uri():
        raise RuntimeError("Installed distribution identity differs from checkout. Restart kernel.")
    report = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "code_revision": git(root, "rev-parse", "HEAD"),
        "package_source": str(source),
        "lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
        "dependency_profile": profile_name(models=models, colab=colab, neural=neural),
        "dependency_lock_applied": not colab,
        "installer": (
            f"pip {importlib.metadata.version('pip')}"
            if importlib.util.find_spec("pip") is not None
            else subprocess.run(
                ["uv", "--version"], check=True, capture_output=True, text=True
            ).stdout.strip()
        ),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("retail-forecast", "numpy", "pandas")
            if any(
                re.sub(r"[-_.]+", "-", candidate).lower() == re.sub(r"[-_.]+", "-", name).lower()
                for candidates in distribution_map.values()
                for candidate in candidates
            )
        },
    }
    if colab:
        report["installation"] = colab_installation
        report["colab_platform_packages"] = {
            distribution: {
                "installed": importlib.metadata.version(distribution),
                "loaded": {
                    name: getattr(module, "__version__", None)
                    for name, module in sys.modules.items()
                    if name.split(".", 1)[0] in distribution_map
                    and distribution in distribution_map.get(name.split(".", 1)[0], [])
                },
            }
            for distribution in ("ipykernel", "jupyter", "setuptools", "traitlets")
            if any(
                re.sub(r"[-_.]+", "-", candidate).lower()
                == re.sub(r"[-_.]+", "-", distribution).lower()
                for distributions in distribution_map.values()
                for candidate in distributions
            )
        }
    print(json.dumps(report, indent=2))
    return report
