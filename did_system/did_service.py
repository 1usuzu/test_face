"""
DID System Integration Service
===============================

Tích hợp DID System với các thành phần khác của dự án:
- Backend API
- Smart Contract
- ZK Proof system
"""

import hashlib
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

from .did_manager import DIDManager, DIDDocument, ServiceEndpoint, DIDMethod
from .key_manager import KeyManager
from .credential_issuer import CredentialIssuer, VerifiableCredential
from .credential_verifier import CredentialVerifier, VerificationResult


class DIDService:
    """
    Main service class for DID operations
    
    Provides a unified interface for:
    - DID management
    - Credential issuance
    - Credential verification
    - Integration with blockchain
    """
    
    def __init__(self, oracle_private_key: Optional[str] = None):
        """
        Initialize DID Service
        
        Args:
            oracle_private_key: Private key for Oracle/Issuer DID
        """
        self.key_manager = KeyManager()
        self.did_manager = DIDManager(self.key_manager)
        
        # Create Oracle/Issuer DID
        if oracle_private_key:
            self.oracle_did, self.oracle_doc, self.oracle_keys = \
                self.did_manager.create_did_from_ethereum(oracle_private_key)
        else:
            self.oracle_did, self.oracle_doc, self.oracle_keys = \
                self.did_manager.create_did(method=DIDMethod.DEEPFAKE)
        
        # Initialize credential services
        self.credential_issuer = CredentialIssuer(
            issuer_did=self.oracle_did,
            key_manager=self.key_manager,
            did_manager=self.did_manager
        )
        
        self.credential_verifier = CredentialVerifier(
            key_manager=self.key_manager,
            did_manager=self.did_manager,
            trusted_issuers=[self.oracle_did]
        )
        
        # Add service endpoint for verification
        self.did_manager.add_service(
            self.oracle_did,
            ServiceEndpoint(
                id=f"{self.oracle_did}#deepfake-verification",
                type="DeepfakeVerificationService",
                service_endpoint="http://localhost:8000/api/verify",
                description="AI-powered deepfake detection service"
            )
        )
    
    # ==================== USER DID OPERATIONS ====================
    
    def create_user_did(self, ethereum_address: str) -> Tuple[str, DIDDocument]:
        """
        Create DID for a user based on their Ethereum address
        
        Args:
            ethereum_address: User's Ethereum wallet address
            
        Returns:
            Tuple of (did, did_document)
        """
        did, doc, keys = self.did_manager.create_did(
            method=DIDMethod.DEEPFAKE,
            ethereum_address=ethereum_address,
            include_eth_key=True
        )
        
        return did, doc
    
    def get_or_create_user_did(self, ethereum_address: str) -> Tuple[str, DIDDocument]:
        """
        Get existing DID or create new one for user
        
        Args:
            ethereum_address: User's Ethereum wallet address
            
        Returns:
            Tuple of (did, did_document)
        """
        # Check if DID exists
        existing_did = self.did_manager.get_did_by_address(ethereum_address)
        if existing_did:
            doc = self.did_manager.resolve(existing_did)
            return existing_did, doc
        
        # Create new DID
        return self.create_user_did(ethereum_address)
    
    def resolve_did(self, did: str) -> Optional[DIDDocument]:
        """Resolve DID to DID Document"""
        return self.did_manager.resolve(did)
    
    def resolve_by_address(self, address: str) -> Optional[DIDDocument]:
        """Resolve DID by Ethereum address"""
        return self.did_manager.resolve_by_address(address)
    
    # ==================== VERIFICATION CREDENTIALS ====================
    
    def issue_verification_credential(
        self,
        user_address: str,
        image_hash: str,
        is_real: bool,
        confidence: float,
        oracle_signature: str
    ) -> VerifiableCredential:
        """
        Issue a Verifiable Credential for deepfake verification
        
        Args:
            user_address: User's Ethereum address
            image_hash: SHA256 hash of verified image
            is_real: Verification result
            confidence: Confidence score (0-1)
            oracle_signature: Oracle's signature
            
        Returns:
            Signed VerifiableCredential
        """
        # Get or create user DID
        user_did, _ = self.get_or_create_user_did(user_address)
        
        # Issue credential
        credential = self.credential_issuer.issue_deepfake_verification_credential(
            subject_did=user_did,
            image_hash=image_hash,
            is_real=is_real,
            confidence=confidence,
            oracle_signature=oracle_signature
        )
        
        return credential
    
    def issue_zk_credential(
        self,
        user_address: str,
        commitment: str,
        nullifier: str,
        proof_verified: bool = True
    ) -> VerifiableCredential:
        """
        Issue a Verifiable Credential for ZK Proof verification
        
        Args:
            user_address: User's Ethereum address
            commitment: ZK proof commitment
            nullifier: ZK proof nullifier
            proof_verified: Whether the proof was verified
            
        Returns:
            Signed VerifiableCredential
        """
        user_did, _ = self.get_or_create_user_did(user_address)
        
        credential = self.credential_issuer.issue_zk_proof_credential(
            subject_did=user_did,
            commitment=commitment,
            nullifier=nullifier,
            proof_data={
                "proofVerified": proof_verified,
                "circuit": "SimpleDeepfakeProof",
                "provingSystem": "Groth16"
            }
        )
        
        return credential
    
    def verify_credential(
        self,
        credential: VerifiableCredential
    ) -> VerificationResult:
        """
        Verify a Verifiable Credential
        
        Args:
            credential: The credential to verify
            
        Returns:
            VerificationResult with status and details
        """
        return self.credential_verifier.verify(
            credential,
            check_revocation=True,
            require_trusted_issuer=True
        )
    
    # ==================== BLOCKCHAIN INTEGRATION ====================
    
    def prepare_did_for_blockchain(self, did: str) -> Dict[str, Any]:
        """
        Prepare DID data for blockchain storage
        
        Args:
            did: The DID to prepare
            
        Returns:
            Dictionary with blockchain-ready data
        """
        doc = self.did_manager.resolve(did)
        if not doc:
            raise ValueError(f"DID not found: {did}")
        
        # Get primary public key
        primary_key = ""
        if doc.verification_method:
            primary_key = doc.verification_method[0].get("publicKeyMultibase", "")
        
        return {
            "did": did,
            "publicKeyBase58": primary_key,
            "controller": doc.controller,
            "created": doc.created,
            "documentHash": hashlib.sha256(doc.to_json().encode()).hexdigest()
        }
    
    def prepare_credential_for_blockchain(
        self,
        credential: VerifiableCredential
    ) -> Dict[str, Any]:
        """
        Prepare credential data for blockchain storage
        
        Returns only essential data to minimize gas costs
        """
        return {
            "credentialId": credential.id,
            "credentialHash": credential.get_hash(),
            "issuer": credential.issuer,
            "subject": credential.credential_subject.get("id", ""),
            "issuanceDate": credential.issuance_date,
            "expirationDate": credential.expiration_date or "",
            "credentialType": credential.type[-1] if credential.type else ""
        }
    
    # ==================== STATISTICS ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall DID system statistics"""
        return {
            "oracle": {
                "did": self.oracle_did,
                "address": self.oracle_keys.get(f"{self.oracle_did}#key-eth-1", {})
            },
            "dids": self.did_manager.get_statistics(),
            "credentials": self.credential_issuer.get_statistics()
        }
