"""
Verifiable Credentials Issuer
=============================

Cấp Verifiable Credentials (VC) cho kết quả Deepfake Verification
theo chuẩn W3C Verifiable Credentials Data Model 1.1

Reference: https://www.w3.org/TR/vc-data-model/
"""

import json
import hashlib
import uuid
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .key_manager import KeyManager
from .did_manager import DIDManager, DIDDocument


class CredentialType(Enum):
    """Types of credentials we issue"""
    DEEPFAKE_VERIFICATION = "DeepfakeVerificationCredential"
    IDENTITY_VERIFICATION = "IdentityVerificationCredential"
    ZK_PROOF_CREDENTIAL = "ZKProofCredential"


@dataclass
class CredentialSubject:
    """The subject of a Verifiable Credential"""
    id: str  # Subject's DID
    verification_result: Optional[Dict[str, Any]] = None
    zk_proof: Optional[Dict[str, Any]] = None
    additional_claims: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"id": self.id}
        
        if self.verification_result:
            result["verificationResult"] = self.verification_result
        if self.zk_proof:
            result["zkProof"] = self.zk_proof
        
        result.update(self.additional_claims)
        return result


@dataclass
class CredentialProof:
    """Proof attached to a Verifiable Credential"""
    type: str  # Ed25519Signature2020, EcdsaSecp256k1Signature2019
    created: str
    verification_method: str  # Key ID used for signing
    proof_purpose: str  # assertionMethod, authentication
    proof_value: str  # The actual signature
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "created": self.created,
            "verificationMethod": self.verification_method,
            "proofPurpose": self.proof_purpose,
            "proofValue": self.proof_value
        }


@dataclass
class VerifiableCredential:
    """
    W3C Verifiable Credential
    
    A credential containing claims about a subject,
    signed by an issuer.
    """
    context: List[str] = field(default_factory=lambda: [
        "https://www.w3.org/2018/credentials/v1",
        "https://www.w3.org/2018/credentials/examples/v1"
    ])
    id: str = ""
    type: List[str] = field(default_factory=lambda: ["VerifiableCredential"])
    issuer: str = ""  # Issuer's DID
    issuance_date: str = ""
    expiration_date: Optional[str] = None
    credential_subject: Dict[str, Any] = field(default_factory=dict)
    proof: Optional[Dict[str, Any]] = None
    
    # Custom fields for our use case
    credential_status: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = f"urn:uuid:{uuid.uuid4()}"
        if not self.issuance_date:
            self.issuance_date = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to W3C VC JSON format"""
        vc = {
            "@context": self.context,
            "id": self.id,
            "type": self.type,
            "issuer": self.issuer,
            "issuanceDate": self.issuance_date,
            "credentialSubject": self.credential_subject
        }
        
        if self.expiration_date:
            vc["expirationDate"] = self.expiration_date
        if self.proof:
            vc["proof"] = self.proof
        if self.credential_status:
            vc["credentialStatus"] = self.credential_status
            
        return vc
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    def get_hash(self) -> str:
        """Get hash of credential (without proof)"""
        vc_dict = self.to_dict()
        vc_dict.pop("proof", None)
        canonical = json.dumps(vc_dict, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode()).hexdigest()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerifiableCredential":
        return cls(
            context=data.get("@context", []),
            id=data.get("id", ""),
            type=data.get("type", []),
            issuer=data.get("issuer", ""),
            issuance_date=data.get("issuanceDate", ""),
            expiration_date=data.get("expirationDate"),
            credential_subject=data.get("credentialSubject", {}),
            proof=data.get("proof"),
            credential_status=data.get("credentialStatus")
        )


class CredentialIssuer:
    """
    Issues Verifiable Credentials for Deepfake Verification results
    
    Features:
    - Issue credentials for verification results
    - Issue ZK proof credentials
    - Revoke credentials
    - Check credential status
    """
    
    def __init__(
        self,
        issuer_did: str,
        key_manager: KeyManager,
        did_manager: DIDManager
    ):
        self.issuer_did = issuer_did
        self.key_manager = key_manager
        self.did_manager = did_manager
        self._issued_credentials: Dict[str, VerifiableCredential] = {}
        self._revoked_credentials: set = set()
    
    # ==================== CREDENTIAL ISSUANCE ====================
    
    def issue_deepfake_verification_credential(
        self,
        subject_did: str,
        image_hash: str,
        is_real: bool,
        confidence: float,
        oracle_signature: str,
        validity_days: int = 365
    ) -> VerifiableCredential:
        """
        Issue a Verifiable Credential for deepfake verification result
        
        Args:
            subject_did: DID of the person/entity being verified
            image_hash: SHA256 hash of the verified image
            is_real: True if image is real, False if deepfake
            confidence: Confidence score (0-1)
            oracle_signature: Signature from the Oracle/AI system
            validity_days: How long the credential is valid
            
        Returns:
            Signed VerifiableCredential
        """
        # Create credential subject
        credential_subject = {
            "id": subject_did,
            "verificationResult": {
                "imageHash": image_hash,
                "isReal": is_real,
                "confidence": confidence,
                "verificationMethod": "EfficientNet-B0-Deepfake-Detector",
                "oracleSignature": oracle_signature
            }
        }
        
        # Calculate expiration
        expiration = datetime.utcnow() + timedelta(days=validity_days)
        
        # Create credential
        vc = VerifiableCredential(
            type=["VerifiableCredential", CredentialType.DEEPFAKE_VERIFICATION.value],
            issuer=self.issuer_did,
            expiration_date=expiration.isoformat() + "Z",
            credential_subject=credential_subject,
            credential_status={
                "id": f"{self.issuer_did}/credentials/status/{uuid.uuid4()}",
                "type": "CredentialStatusList2021"
            }
        )
        
        # Sign the credential
        signed_vc = self._sign_credential(vc)
        
        # Store
        self._issued_credentials[signed_vc.id] = signed_vc
        
        return signed_vc
    
    def issue_zk_proof_credential(
        self,
        subject_did: str,
        commitment: str,
        nullifier: str,
        proof_data: Dict[str, Any],
        validity_days: int = 365
    ) -> VerifiableCredential:
        """
        Issue a Verifiable Credential for ZK Proof verification
        
        This credential proves that a ZK proof was verified without
        revealing the underlying data.
        
        Args:
            subject_did: DID of the prover
            commitment: ZK proof commitment
            nullifier: ZK proof nullifier
            proof_data: Additional proof metadata
            validity_days: Credential validity period
            
        Returns:
            Signed VerifiableCredential
        """
        credential_subject = {
            "id": subject_did,
            "zkProof": {
                "type": "Groth16",
                "commitment": commitment,
                "nullifier": nullifier,
                "verified": True,
                "verifiedAt": datetime.utcnow().isoformat() + "Z",
                **proof_data
            }
        }
        
        expiration = datetime.utcnow() + timedelta(days=validity_days)
        
        vc = VerifiableCredential(
            type=["VerifiableCredential", CredentialType.ZK_PROOF_CREDENTIAL.value],
            issuer=self.issuer_did,
            expiration_date=expiration.isoformat() + "Z",
            credential_subject=credential_subject
        )
        
        signed_vc = self._sign_credential(vc)
        self._issued_credentials[signed_vc.id] = signed_vc
        
        return signed_vc
    
    # ==================== SIGNING ====================
    
    def _sign_credential(self, vc: VerifiableCredential) -> VerifiableCredential:
        """
        Sign a Verifiable Credential
        
        Uses Ed25519 or secp256k1 depending on available keys
        """
        # Get signing key
        issuer_doc = self.did_manager.resolve(self.issuer_did)
        if not issuer_doc:
            raise ValueError(f"Issuer DID not found: {self.issuer_did}")
        
        # Find assertion method key
        if not issuer_doc.assertion_method:
            raise ValueError("Issuer has no assertion method keys")
        
        key_id = issuer_doc.assertion_method[0]
        key = self.key_manager.get_key(key_id)
        
        if not key:
            raise ValueError(f"Key not found: {key_id}")
        
        # Create proof
        created = datetime.utcnow().isoformat() + "Z"
        
        # Get credential hash for signing
        vc_hash = vc.get_hash()
        
        # Sign based on key type
        if key.key_type == "Ed25519VerificationKey2020":
            signature = self.key_manager.sign_ed25519(key_id, vc_hash.encode())
            proof_type = "Ed25519Signature2020"
        else:
            signature = self.key_manager.sign_secp256k1(key_id, vc_hash)
            proof_type = "EcdsaSecp256k1Signature2019"
        
        proof = CredentialProof(
            type=proof_type,
            created=created,
            verification_method=key_id,
            proof_purpose="assertionMethod",
            proof_value=signature
        )
        
        vc.proof = proof.to_dict()
        return vc
    
    # ==================== REVOCATION ====================
    
    def revoke_credential(self, credential_id: str, reason: str = "") -> bool:
        """
        Revoke a credential
        
        Args:
            credential_id: The credential ID to revoke
            reason: Reason for revocation
            
        Returns:
            True if revoked successfully
        """
        if credential_id not in self._issued_credentials:
            return False
        
        self._revoked_credentials.add(credential_id)
        return True
    
    def is_revoked(self, credential_id: str) -> bool:
        """Check if credential is revoked"""
        return credential_id in self._revoked_credentials
    
    # ==================== UTILITIES ====================
    
    def get_credential(self, credential_id: str) -> Optional[VerifiableCredential]:
        """Get credential by ID"""
        return self._issued_credentials.get(credential_id)
    
    def list_credentials(self, subject_did: Optional[str] = None) -> List[VerifiableCredential]:
        """List issued credentials, optionally filtered by subject"""
        credentials = list(self._issued_credentials.values())
        
        if subject_did:
            credentials = [
                vc for vc in credentials 
                if vc.credential_subject.get("id") == subject_did
            ]
        
        return credentials
    
    def get_statistics(self) -> Dict[str, int]:
        """Get issuer statistics"""
        return {
            "total_issued": len(self._issued_credentials),
            "total_revoked": len(self._revoked_credentials),
            "active": len(self._issued_credentials) - len(self._revoked_credentials)
        }
