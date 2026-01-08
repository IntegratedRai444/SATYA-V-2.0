"""
Proof of Analysis Generation and Validation

This module provides functionality to generate and validate cryptographic proofs
for analysis results to ensure integrity and non-repudiation.
"""

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

# Use a secure secret key for HMAC (in production, load from secure storage)
PROOF_SECRET = b"satya-secure-proof-key-2025"


@dataclass
class ProofOfAnalysis:
    """Proof of Analysis data structure"""

    model_name: str
    model_version: str
    modality: str  # 'image', 'video', 'audio', 'webcam'
    timestamp: float
    inference_duration: float
    frames_analyzed: int = 1
    metadata: Optional[Dict[str, Any]] = None
    signature: Optional[bytes] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert proof to dictionary for serialization"""
        data = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "modality": self.modality,
            "timestamp": self.timestamp,
            "inference_duration": self.inference_duration,
            "frames_analyzed": self.frames_analyzed,
            "metadata": self.metadata or {},
        }

        if self.signature:
            data["signature"] = base64.b64encode(self.signature).decode("utf-8")

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProofOfAnalysis":
        """Create ProofOfAnalysis from dictionary"""
        signature = None
        if "signature" in data:
            signature = base64.b64decode(data["signature"])

        return cls(
            model_name=data["model_name"],
            model_version=data["model_version"],
            modality=data["modality"],
            timestamp=data["timestamp"],
            inference_duration=data["inference_duration"],
            frames_analyzed=data.get("frames_analyzed", 1),
            metadata=data.get("metadata"),
            signature=signature,
        )

    def sign(self, private_key: Optional[bytes] = None) -> None:
        """Sign the proof data"""
        # In a real implementation, use proper asymmetric cryptography
        # This is a simplified version using HMAC for demonstration
        data = json.dumps(
            {
                "model_name": self.model_name,
                "model_version": self.model_version,
                "modality": self.modality,
                "timestamp": self.timestamp,
                "inference_duration": self.inference_duration,
                "frames_analyzed": self.frames_analyzed,
                "metadata": self.metadata or {},
            },
            sort_keys=True,
        ).encode("utf-8")

        self.signature = hmac.new(PROOF_SECRET, data, hashlib.sha256).digest()

    def verify(self) -> bool:
        """Verify the proof signature"""
        if not self.signature:
            return False

        # Recreate the signature with the same data
        data = json.dumps(
            {
                "model_name": self.model_name,
                "model_version": self.model_version,
                "modality": self.modality,
                "timestamp": self.timestamp,
                "inference_duration": self.inference_duration,
                "frames_analyzed": self.frames_analyzed,
                "metadata": self.metadata or {},
            },
            sort_keys=True,
        ).encode("utf-8")

        expected_signature = hmac.new(PROOF_SECRET, data, hashlib.sha256).digest()
        return hmac.compare_digest(self.signature, expected_signature)

    def is_fresh(self, max_age_seconds: int = 300) -> bool:
        """Check if the proof is still fresh"""
        current_time = time.time()
        return (current_time - self.timestamp) <= max_age_seconds


def generate_proof(
    model_name: str,
    model_version: str,
    modality: str,
    inference_duration: float,
    frames_analyzed: int = 1,
    metadata: Optional[Dict[str, Any]] = None,
) -> ProofOfAnalysis:
    """Generate a new proof of analysis"""
    proof = ProofOfAnalysis(
        model_name=model_name,
        model_version=model_version,
        modality=modality,
        timestamp=time.time(),
        inference_duration=inference_duration,
        frames_analyzed=frames_analyzed,
        metadata=metadata or {},
    )

    proof.sign()
    return proof


def verify_proof(proof_data: Dict[str, Any]) -> Tuple[bool, str]:
    """Verify a proof of analysis"""
    try:
        proof = ProofOfAnalysis.from_dict(proof_data)

        # Verify the signature
        if not proof.verify():
            return False, "Invalid proof signature"

        # Check if the proof is fresh (e.g., not older than 5 minutes)
        if not proof.is_fresh():
            return False, "Proof has expired"

        return True, "Proof is valid"

    except (KeyError, ValueError, TypeError) as e:
        return False, f"Invalid proof format: {str(e)}"
