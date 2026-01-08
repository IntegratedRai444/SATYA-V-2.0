"""
Model versioning utilities for SatyaAI.
Provides functionality to manage model versions and compatibility.
"""
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import semver

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Class to represent a model version."""

    version: str
    path: str
    description: str = ""
    is_active: bool = True
    created_at: str = ""
    compatibility: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, any]] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if self.compatibility is None:
            self.compatibility = {}
        if self.metadata is None:
            self.metadata = {}


class ModelVersionManager:
    """Manages model versions and their metadata."""

    def __init__(self, models_dir: str):
        """
        Initialize the version manager.

        Args:
            models_dir: Base directory containing all models
        """
        self.models_dir = Path(models_dir)
        self.versions_file = self.models_dir / "model_versions.json"
        self.versions = self._load_versions()

    def _load_versions(self) -> Dict[str, List[ModelVersion]]:
        """Load version information from disk."""
        if not self.versions_file.exists():
            return {}

        try:
            with open(self.versions_file, "r") as f:
                data = json.load(f)

            versions = {}
            for model_name, model_versions in data.items():
                versions[model_name] = [
                    ModelVersion(**version_data) for version_data in model_versions
                ]
            return versions

        except Exception as e:
            logger.error(f"Error loading model versions: {e}")
            return {}

    def _save_versions(self):
        """Save version information to disk."""
        try:
            # Convert ModelVersion objects to dictionaries
            data = {
                model_name: [asdict(version) for version in versions]
                for model_name, versions in self.versions.items()
            }

            with open(self.versions_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving model versions: {e}")
            raise

    def add_version(
        self,
        model_name: str,
        version: str,
        model_path: str,
        description: str = "",
        compatibility: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, any]] = None,
        is_active: bool = True,
    ) -> ModelVersion:
        """
        Add a new model version.

        Args:
            model_name: Name of the model
            version: Version string (semver)
            model_path: Path to the model file
            description: Description of this version
            compatibility: Compatibility requirements
            metadata: Additional metadata
            is_active: Whether this version should be active

        Returns:
            The created ModelVersion instance
        """
        if not self._is_valid_semver(version):
            raise ValueError(f"Invalid semantic version: {version}")

        version_obj = ModelVersion(
            version=version,
            path=str(model_path),
            description=description,
            is_active=is_active,
            compatibility=compatibility or {},
            metadata=metadata or {},
        )

        if model_name not in self.versions:
            self.versions[model_name] = []

        self.versions[model_name].append(version_obj)
        self._save_versions()

        return version_obj

    def get_latest_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get the latest version of a model."""
        if model_name not in self.versions or not self.versions[model_name]:
            return None

        # Sort by version number (newest first)
        versions = sorted(
            self.versions[model_name],
            key=lambda v: semver.VersionInfo.parse(v.version),
            reverse=True,
        )

        # Return the first active version
        for version in versions:
            if version.is_active:
                return version

        return None

    def get_version(self, model_name: str, version: str) -> Optional[ModelVersion]:
        """Get a specific version of a model."""
        if model_name not in self.versions:
            return None

        for v in self.versions[model_name]:
            if v.version == version:
                return v

        return None

    def deactivate_version(self, model_name: str, version: str) -> bool:
        """Mark a specific version as inactive."""
        version_obj = self.get_version(model_name, version)
        if not version_obj:
            return False

        version_obj.is_active = False
        self._save_versions()
        return True

    @staticmethod
    def _is_valid_semver(version: str) -> bool:
        """Check if a string is a valid semantic version."""
        try:
            semver.VersionInfo.parse(version)
            return True
        except ValueError:
            return False

    def check_compatibility(
        self, model_name: str, version: str, requirements: Dict[str, str]
    ) -> Tuple[bool, Dict[str, str]]:
        """
        Check if a model version is compatible with the given requirements.

        Args:
            model_name: Name of the model
            version: Version string
            requirements: Dictionary of requirements (e.g., {'python': '>=3.8'})

        Returns:
            Tuple of (is_compatible, details)
        """
        version_obj = self.get_version(model_name, version)
        if not version_obj:
            return False, {"error": f"Version {version} of {model_name} not found"}

        if not version_obj.is_active:
            return False, {"error": f"Version {version} of {model_name} is inactive"}

        # Check compatibility requirements
        results = {}
        all_ok = True

        for req_key, req_value in (version_obj.compatibility or {}).items():
            if req_key in requirements:
                try:
                    # Simple version comparison
                    # For production, consider using packaging.version or similar
                    if not self._check_version_requirement(
                        requirements[req_key], req_value
                    ):
                        results[
                            req_key
                        ] = f"Requires {req_key} {req_value}, but got {requirements[req_key]}"
                        all_ok = False
                except Exception as e:
                    logger.warning(f"Error checking {req_key} compatibility: {e}")
                    results[req_key] = f"Error checking compatibility: {str(e)}"
                    all_ok = False

        return all_ok, results

    @staticmethod
    def _check_version_requirement(requirement: str, version: str) -> bool:
        """
        Check if a version satisfies the requirement.

        Args:
            requirement: Version requirement (e.g., '>=1.2.3')
            version: Version to check

        Returns:
            bool: True if version satisfies the requirement
        """
        # Simple implementation - for production, use packaging.version or similar
        req_ver = semver.VersionInfo.parse(requirement.lstrip("=<>!~^"))
        ver = semver.VersionInfo.parse(version)

        if requirement.startswith(">="):
            return ver >= req_ver
        elif requirement.startswith(">"):
            return ver > req_ver
        elif requirement.startswith("<="):
            return ver <= req_ver
        elif requirement.startswith("<"):
            return ver < req_ver
        elif requirement.startswith("=="):
            return ver == req_ver
        elif requirement.startswith("!="):
            return ver != req_ver
        else:
            # Default to exact match
            return str(ver) == str(req_ver)
