"""
Decentralized Identity (DID) System
====================================

Hệ thống định danh phi tập trung theo chuẩn W3C DID Core 1.0

Components:
- DIDManager: Tạo và quản lý DID Documents
- CredentialIssuer: Cấp Verifiable Credentials
- CredentialVerifier: Xác thực Verifiable Credentials
- KeyManager: Quản lý cryptographic keys
- DIDService: Service tích hợp chính

Standards:
- W3C DID Core 1.0: https://www.w3.org/TR/did-core/
- W3C Verifiable Credentials: https://www.w3.org/TR/vc-data-model/
"""

from .did_manager import DIDManager, DIDDocument, DIDMethod, ServiceEndpoint
from .credential_issuer import CredentialIssuer, VerifiableCredential, CredentialType
from .credential_verifier import CredentialVerifier, VerificationResult, VerificationStatus
from .key_manager import KeyManager, KeyPair
from .did_service import DIDService

__version__ = "1.0.0"
__all__ = [
    # Core DID
    "DIDManager",
    "DIDDocument",
    "DIDMethod",
    "ServiceEndpoint",
    
    # Keys
    "KeyManager",
    "KeyPair",
    
    # Credentials
    "CredentialIssuer",
    "CredentialVerifier",
    "VerifiableCredential",
    "VerificationResult",
    "VerificationStatus",
    "CredentialType",
    
    # Service
    "DIDService"
]
