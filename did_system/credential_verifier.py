"""
Verifiable Credentials Verifier
================================

Xác thực Verifiable Credentials theo chuẩn W3C

Features:
- Verify credential signatures
- Check expiration
- Check revocation status
- Validate credential structure
"""

import json
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .key_manager import KeyManager
from .did_manager import DIDManager
from .credential_issuer import VerifiableCredential, CredentialIssuer


class VerificationStatus(Enum):
    """Credential verification status"""
    VALID = "valid"
    INVALID_SIGNATURE = "invalid_signature"
    EXPIRED = "expired"
    REVOKED = "revoked"
    ISSUER_NOT_FOUND = "issuer_not_found"
    KEY_NOT_FOUND = "key_not_found"
    MALFORMED = "malformed"
    NOT_YET_VALID = "not_yet_valid"


@dataclass
class VerificationResult:
    """Result of credential verification"""
    status: VerificationStatus
    credential_id: str
    issuer: str
    subject: str
    is_valid: bool
    checks: Dict[str, bool]
    errors: List[str]
    verified_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "credentialId": self.credential_id,
            "issuer": self.issuer,
            "subject": self.subject,
            "isValid": self.is_valid,
            "checks": self.checks,
            "errors": self.errors,
            "verifiedAt": self.verified_at
        }


class CredentialVerifier:
    """
    Verifies Verifiable Credentials
    
    Performs the following checks:
    1. Structure validation
    2. Signature verification
    3. Expiration check
    4. Revocation check
    5. Issuer trust check
    """
    
    def __init__(
        self,
        key_manager: KeyManager,
        did_manager: DIDManager,
        trusted_issuers: Optional[List[str]] = None
    ):
        self.key_manager = key_manager
        self.did_manager = did_manager
        self.trusted_issuers = set(trusted_issuers or [])
        self._revocation_lists: Dict[str, set] = {}
    
    # ==================== VERIFICATION ====================
    
    def verify(
        self,
        credential: VerifiableCredential,
        check_revocation: bool = True,
        require_trusted_issuer: bool = False
    ) -> VerificationResult:
        """
        Verify a Verifiable Credential
        
        Args:
            credential: The credential to verify
            check_revocation: Whether to check revocation status
            require_trusted_issuer: Whether issuer must be in trusted list
            
        Returns:
            VerificationResult with detailed status
        """
        errors = []
        checks = {
            "structure": False,
            "signature": False,
            "expiration": False,
            "revocation": False,
            "trusted_issuer": False
        }
        
        now = datetime.utcnow()
        
        # 1. Structure validation
        structure_valid, structure_errors = self._validate_structure(credential)
        checks["structure"] = structure_valid
        errors.extend(structure_errors)
        
        if not structure_valid:
            return self._create_result(
                VerificationStatus.MALFORMED,
                credential,
                checks,
                errors
            )
        
        # 2. Check expiration
        if credential.expiration_date:
            try:
                expiration = datetime.fromisoformat(
                    credential.expiration_date.replace("Z", "+00:00")
                ).replace(tzinfo=None)
                
                if now > expiration:
                    checks["expiration"] = False
                    errors.append("Credential has expired")
                    return self._create_result(
                        VerificationStatus.EXPIRED,
                        credential,
                        checks,
                        errors
                    )
                else:
                    checks["expiration"] = True
            except ValueError as e:
                errors.append(f"Invalid expiration date format: {e}")
        else:
            checks["expiration"] = True  # No expiration = always valid
        
        # 3. Verify signature
        sig_valid, sig_error = self._verify_signature(credential)
        checks["signature"] = sig_valid
        
        if not sig_valid:
            if sig_error:
                errors.append(sig_error)
            return self._create_result(
                VerificationStatus.INVALID_SIGNATURE,
                credential,
                checks,
                errors
            )
        
        # 4. Check revocation
        if check_revocation:
            is_revoked = self._check_revocation(credential)
            checks["revocation"] = not is_revoked
            
            if is_revoked:
                errors.append("Credential has been revoked")
                return self._create_result(
                    VerificationStatus.REVOKED,
                    credential,
                    checks,
                    errors
                )
        else:
            checks["revocation"] = True
        
        # 5. Check trusted issuer
        if require_trusted_issuer:
            is_trusted = credential.issuer in self.trusted_issuers
            checks["trusted_issuer"] = is_trusted
            
            if not is_trusted:
                errors.append(f"Issuer {credential.issuer} is not trusted")
        else:
            checks["trusted_issuer"] = True
        
        # All checks passed
        return self._create_result(
            VerificationStatus.VALID,
            credential,
            checks,
            errors
        )
    
    def verify_json(self, credential_json: str, **kwargs) -> VerificationResult:
        """Verify credential from JSON string"""
        try:
            data = json.loads(credential_json)
            credential = VerifiableCredential.from_dict(data)
            return self.verify(credential, **kwargs)
        except json.JSONDecodeError as e:
            return VerificationResult(
                status=VerificationStatus.MALFORMED,
                credential_id="",
                issuer="",
                subject="",
                is_valid=False,
                checks={},
                errors=[f"Invalid JSON: {e}"],
                verified_at=datetime.utcnow().isoformat() + "Z"
            )
    
    # ==================== VALIDATION HELPERS ====================
    
    def _validate_structure(self, credential: VerifiableCredential) -> Tuple[bool, List[str]]:
        """Validate credential structure"""
        errors = []
        
        # Required fields
        if not credential.id:
            errors.append("Missing credential ID")
        
        if not credential.type or "VerifiableCredential" not in credential.type:
            errors.append("Invalid or missing credential type")
        
        if not credential.issuer:
            errors.append("Missing issuer")
        
        if not credential.issuance_date:
            errors.append("Missing issuance date")
        
        if not credential.credential_subject:
            errors.append("Missing credential subject")
        
        if not credential.proof:
            errors.append("Missing proof")
        
        return len(errors) == 0, errors
    
    def _verify_signature(self, credential: VerifiableCredential) -> Tuple[bool, Optional[str]]:
        """Verify credential signature"""
        if not credential.proof:
            return False, "No proof present"
        
        proof = credential.proof
        
        # Get verification method (key)
        verification_method = proof.get("verificationMethod")
        if not verification_method:
            return False, "No verification method in proof"
        
        # Get credential hash
        vc_hash = credential.get_hash()
        
        # Get signature
        proof_value = proof.get("proofValue")
        if not proof_value:
            return False, "No proof value"
        
        # Resolve issuer DID to get public key
        issuer_doc = self.did_manager.resolve(credential.issuer)
        if not issuer_doc:
            return False, f"Could not resolve issuer DID: {credential.issuer}"
        
        # Find the verification method
        key_data = None
        for vm in issuer_doc.verification_method:
            if vm.get("id") == verification_method:
                key_data = vm
                break
        
        if not key_data:
            return False, f"Verification method not found: {verification_method}"
        
        # Verify based on key type
        key_type = key_data.get("type")
        public_key = key_data.get("publicKeyMultibase", "").lstrip("z")
        
        try:
            if key_type == "Ed25519VerificationKey2020":
                is_valid = self.key_manager.verify_ed25519(
                    public_key,
                    vc_hash.encode(),
                    proof_value
                )
            elif key_type == "EcdsaSecp256k1VerificationKey2019":
                is_valid = self.key_manager.verify_secp256k1(
                    vc_hash,
                    proof_value,
                    public_key
                )
            else:
                return False, f"Unsupported key type: {key_type}"
            
            return is_valid, None if is_valid else "Signature verification failed"
            
        except Exception as e:
            return False, f"Signature verification error: {e}"
    
    def _check_revocation(self, credential: VerifiableCredential) -> bool:
        """Check if credential is revoked"""
        # Check local revocation list
        issuer = credential.issuer
        if issuer in self._revocation_lists:
            if credential.id in self._revocation_lists[issuer]:
                return True
        
        # TODO: Check on-chain revocation status
        # TODO: Check credential status endpoint
        
        return False
    
    def _create_result(
        self,
        status: VerificationStatus,
        credential: VerifiableCredential,
        checks: Dict[str, bool],
        errors: List[str]
    ) -> VerificationResult:
        """Create verification result"""
        subject = credential.credential_subject.get("id", "") if credential.credential_subject else ""
        
        return VerificationResult(
            status=status,
            credential_id=credential.id,
            issuer=credential.issuer,
            subject=subject,
            is_valid=status == VerificationStatus.VALID,
            checks=checks,
            errors=errors,
            verified_at=datetime.utcnow().isoformat() + "Z"
        )
    
    # ==================== TRUST MANAGEMENT ====================
    
    def add_trusted_issuer(self, issuer_did: str):
        """Add issuer to trusted list"""
        self.trusted_issuers.add(issuer_did)
    
    def remove_trusted_issuer(self, issuer_did: str):
        """Remove issuer from trusted list"""
        self.trusted_issuers.discard(issuer_did)
    
    def is_trusted_issuer(self, issuer_did: str) -> bool:
        """Check if issuer is trusted"""
        return issuer_did in self.trusted_issuers
    
    # ==================== REVOCATION MANAGEMENT ====================
    
    def add_revocation(self, issuer_did: str, credential_id: str):
        """Add credential to revocation list"""
        if issuer_did not in self._revocation_lists:
            self._revocation_lists[issuer_did] = set()
        self._revocation_lists[issuer_did].add(credential_id)
    
    def sync_revocation_list(self, issuer_did: str, revoked_ids: List[str]):
        """Sync revocation list from issuer"""
        self._revocation_lists[issuer_did] = set(revoked_ids)
