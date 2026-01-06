"""
DID System Tests
=================

Kiểm thử toàn diện cho DID System
"""

# pytest is optional
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

import json
import hashlib
from datetime import datetime, timedelta

# Add parent directory to path for imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from did_system.key_manager import KeyManager, KeyPair
from did_system.did_manager import DIDManager, DIDDocument, DIDMethod
from did_system.credential_issuer import CredentialIssuer, VerifiableCredential
from did_system.credential_verifier import CredentialVerifier, VerificationStatus
from did_system.did_service import DIDService


class TestKeyManager:
    """Test KeyManager functionality"""
    
    def setup_method(self):
        self.key_manager = KeyManager()
    
    def test_generate_ed25519(self):
        """Test Ed25519 key generation"""
        key = self.key_manager.generate_ed25519("test-key-1")
        
        assert key is not None
        assert key.key_type == "Ed25519VerificationKey2020"
        assert len(key.private_key) > 0
        assert len(key.public_key) > 0
        print(f"✅ Ed25519 key generated: {key.key_id}")
    
    def test_generate_secp256k1(self):
        """Test secp256k1 key generation"""
        key = self.key_manager.generate_secp256k1("test-key-2")
        
        assert key is not None
        assert key.key_type == "EcdsaSecp256k1VerificationKey2019"
        assert key.ethereum_address is not None
        assert key.ethereum_address.startswith("0x")
        print(f"✅ secp256k1 key generated with address: {key.ethereum_address}")
    
    def test_sign_and_verify_ed25519(self):
        """Test Ed25519 signing and verification"""
        key = self.key_manager.generate_ed25519("sign-test")
        message = b"Test message for signing"
        
        signature = self.key_manager.sign_ed25519("sign-test", message)
        assert signature is not None
        
        is_valid = self.key_manager.verify_ed25519(
            key.public_key, message, signature
        )
        assert is_valid == True
        print(f"✅ Ed25519 sign/verify: Valid")
    
    def test_sign_and_verify_secp256k1(self):
        """Test secp256k1 signing and verification"""
        key = self.key_manager.generate_secp256k1("secp-sign-test")
        message = "Test message for secp256k1"
        
        signature = self.key_manager.sign_secp256k1("secp-sign-test", message)
        assert signature is not None
        
        # Use ethereum_address for verification (includes 0x prefix)
        is_valid = self.key_manager.verify_secp256k1(
            message, signature, key.ethereum_address
        )
        assert is_valid == True
        print(f"✅ secp256k1 sign/verify: Valid")


class TestDIDManager:
    """Test DIDManager functionality"""
    
    def setup_method(self):
        self.key_manager = KeyManager()
        self.did_manager = DIDManager(self.key_manager)
    
    def test_create_did_key(self):
        """Test did:key creation"""
        did, doc, keys = self.did_manager.create_did(method=DIDMethod.KEY)
        
        assert did.startswith("did:key:")
        assert doc is not None
        assert len(doc.verification_method) > 0
        print(f"✅ Created did:key: {did[:50]}...")
    
    def test_create_did_deepfake(self):
        """Test did:deepfake creation"""
        did, doc, keys = self.did_manager.create_did(method=DIDMethod.DEEPFAKE)
        
        assert did.startswith("did:deepfake:")
        assert doc is not None
        print(f"✅ Created did:deepfake: {did}")
    
    def test_create_did_with_ethereum(self):
        """Test DID creation with Ethereum address"""
        eth_address = "0x742d35Cc6634C0532925a3b844Bc9e7595f8c1F5"
        did, doc, keys = self.did_manager.create_did(
            method=DIDMethod.DEEPFAKE,
            ethereum_address=eth_address,
            include_eth_key=True
        )
        
        assert did.startswith("did:deepfake:")
        assert "EcdsaSecp256k1" in str(doc.verification_method)
        print(f"✅ Created DID with ETH: {did}")
    
    def test_resolve_did(self):
        """Test DID resolution"""
        did, doc, _ = self.did_manager.create_did(method=DIDMethod.DEEPFAKE)
        
        resolved = self.did_manager.resolve(did)
        
        assert resolved is not None
        assert resolved.id == did
        print(f"✅ Resolved DID: {did}")
    
    def test_update_did(self):
        """Test DID document update"""
        did, doc, _ = self.did_manager.create_did(method=DIDMethod.DEEPFAKE)
        
        initial_count = len(doc.verification_method)
        
        # Add new verification method
        new_key = self.key_manager.generate_ed25519(f"{did}#key-2")
        
        updated = self.did_manager.update_did(did, add_verification_methods=[{
            "id": f"{did}#key-2",
            "type": "Ed25519VerificationKey2020",
            "controller": did,
            "publicKeyMultibase": f"z{new_key.public_key}"
        }])
        
        assert updated == True
        resolved = self.did_manager.resolve(did)
        assert len(resolved.verification_method) == initial_count + 1
        print(f"✅ Updated DID with new key")
    
    def test_deactivate_did(self):
        """Test DID deactivation"""
        did, doc, _ = self.did_manager.create_did(method=DIDMethod.DEEPFAKE)
        
        result = self.did_manager.deactivate(did)
        assert result == True
        
        resolved = self.did_manager.resolve(did)
        assert resolved is None
        print(f"✅ Deactivated DID: {did}")


class TestCredentialIssuer:
    """Test CredentialIssuer functionality"""
    
    def setup_method(self):
        self.key_manager = KeyManager()
        self.did_manager = DIDManager(self.key_manager)
        
        # Create issuer DID
        self.issuer_did, self.issuer_doc, _ = self.did_manager.create_did(
            method=DIDMethod.DEEPFAKE
        )
        
        self.issuer = CredentialIssuer(
            issuer_did=self.issuer_did,
            key_manager=self.key_manager,
            did_manager=self.did_manager
        )
        
        # Create subject DID
        self.subject_did, _, _ = self.did_manager.create_did(method=DIDMethod.DEEPFAKE)
    
    def test_issue_deepfake_credential(self):
        """Test issuing deepfake verification credential"""
        image_hash = hashlib.sha256(b"test image data").hexdigest()
        
        credential = self.issuer.issue_deepfake_verification_credential(
            subject_did=self.subject_did,
            image_hash=image_hash,
            is_real=True,
            confidence=0.95,
            oracle_signature="0x" + "a" * 130
        )
        
        assert credential is not None
        assert credential.issuer == self.issuer_did
        assert credential.credential_subject["id"] == self.subject_did
        assert credential.proof is not None
        print(f"✅ Issued credential: {credential.id}")
    
    def test_issue_zk_credential(self):
        """Test issuing ZK proof credential"""
        credential = self.issuer.issue_zk_proof_credential(
            subject_did=self.subject_did,
            commitment="0x" + "b" * 64,
            nullifier="0x" + "c" * 64,
            proof_data={"circuitId": "SimpleDeepfakeProof"}
        )
        
        assert credential is not None
        assert "ZKProofCredential" in credential.type
        assert credential.credential_subject["zkProof"]["verified"] == True
        print(f"✅ Issued ZK credential: {credential.id}")
    
    def test_credential_structure(self):
        """Test credential follows W3C structure"""
        image_hash = hashlib.sha256(b"test").hexdigest()
        
        credential = self.issuer.issue_deepfake_verification_credential(
            subject_did=self.subject_did,
            image_hash=image_hash,
            is_real=True,
            confidence=0.9,
            oracle_signature="0xsig"
        )
        
        vc_dict = credential.to_dict()
        
        # Check required fields
        assert "@context" in vc_dict
        assert "id" in vc_dict
        assert "type" in vc_dict
        assert "issuer" in vc_dict
        assert "issuanceDate" in vc_dict
        assert "credentialSubject" in vc_dict
        assert "proof" in vc_dict
        
        # Verify context includes W3C
        assert "https://www.w3.org/2018/credentials/v1" in vc_dict["@context"]
        
        print(f"✅ Credential structure valid")
        print(f"   - Context: {len(vc_dict['@context'])} entries")
        print(f"   - Type: {vc_dict['type']}")
    
    def test_revoke_credential(self):
        """Test credential revocation"""
        image_hash = hashlib.sha256(b"revoke test").hexdigest()
        
        credential = self.issuer.issue_deepfake_verification_credential(
            subject_did=self.subject_did,
            image_hash=image_hash,
            is_real=True,
            confidence=0.85,
            oracle_signature="0xsig"
        )
        
        # Revoke
        result = self.issuer.revoke_credential(credential.id, "Test revocation")
        assert result == True
        
        # Check revocation
        is_revoked = self.issuer.is_revoked(credential.id)
        assert is_revoked == True
        
        print(f"✅ Credential revoked: {credential.id}")


class TestCredentialVerifier:
    """Test CredentialVerifier functionality"""
    
    def setup_method(self):
        self.key_manager = KeyManager()
        self.did_manager = DIDManager(self.key_manager)
        
        # Create issuer
        self.issuer_did, _, _ = self.did_manager.create_did(method=DIDMethod.DEEPFAKE)
        self.issuer = CredentialIssuer(
            issuer_did=self.issuer_did,
            key_manager=self.key_manager,
            did_manager=self.did_manager
        )
        
        # Create verifier
        self.verifier = CredentialVerifier(
            key_manager=self.key_manager,
            did_manager=self.did_manager,
            trusted_issuers=[self.issuer_did]
        )
        
        # Create subject
        self.subject_did, _, _ = self.did_manager.create_did(method=DIDMethod.DEEPFAKE)
    
    def test_verify_valid_credential(self):
        """Test verifying a valid credential"""
        image_hash = hashlib.sha256(b"valid test").hexdigest()
        
        credential = self.issuer.issue_deepfake_verification_credential(
            subject_did=self.subject_did,
            image_hash=image_hash,
            is_real=True,
            confidence=0.92,
            oracle_signature="0xsig"
        )
        
        result = self.verifier.verify(credential)
        
        assert result.is_valid == True
        assert result.status == VerificationStatus.VALID
        print(f"✅ Valid credential verified")
        print(f"   - Checks: {result.checks}")
    
    def test_verify_expired_credential(self):
        """Test verifying an expired credential"""
        image_hash = hashlib.sha256(b"expired test").hexdigest()
        
        credential = self.issuer.issue_deepfake_verification_credential(
            subject_did=self.subject_did,
            image_hash=image_hash,
            is_real=True,
            confidence=0.9,
            oracle_signature="0xsig",
            validity_days=0  # Immediate expiry
        )
        
        # Manually set expiration to past
        past = datetime.utcnow() - timedelta(days=1)
        credential.expiration_date = past.isoformat() + "Z"
        
        result = self.verifier.verify(credential)
        
        assert result.is_valid == False
        assert result.status == VerificationStatus.EXPIRED
        print(f"✅ Expired credential detected")
    
    def test_verify_malformed_credential(self):
        """Test verifying a malformed credential"""
        credential = VerifiableCredential(
            type=[],  # Missing "VerifiableCredential"
            issuer="",
            credential_subject={}
        )
        
        result = self.verifier.verify(credential)
        
        assert result.is_valid == False
        assert result.status == VerificationStatus.MALFORMED
        assert len(result.errors) > 0
        print(f"✅ Malformed credential detected")
        print(f"   - Errors: {result.errors}")
    
    def test_trusted_issuer_check(self):
        """Test trusted issuer verification"""
        # Create untrusted issuer
        untrusted_did, _, _ = self.did_manager.create_did(method=DIDMethod.DEEPFAKE)
        untrusted_issuer = CredentialIssuer(
            issuer_did=untrusted_did,
            key_manager=self.key_manager,
            did_manager=self.did_manager
        )
        
        image_hash = hashlib.sha256(b"untrusted test").hexdigest()
        credential = untrusted_issuer.issue_deepfake_verification_credential(
            subject_did=self.subject_did,
            image_hash=image_hash,
            is_real=True,
            confidence=0.9,
            oracle_signature="0xsig"
        )
        
        result = self.verifier.verify(credential, require_trusted_issuer=True)
        
        assert result.checks["trusted_issuer"] == False
        print(f"✅ Untrusted issuer detected")


class TestDIDService:
    """Test DIDService integration"""
    
    def setup_method(self):
        self.service = DIDService()
    
    def test_service_initialization(self):
        """Test service initializes properly"""
        assert self.service.oracle_did is not None
        assert self.service.oracle_did.startswith("did:deepfake:")
        print(f"✅ Service initialized with Oracle DID: {self.service.oracle_did}")
    
    def test_create_user_did(self):
        """Test user DID creation"""
        eth_address = "0x742d35Cc6634C0532925a3b844Bc9e7595f8c1F5"
        
        did, doc = self.service.create_user_did(eth_address)
        
        assert did is not None
        assert doc is not None
        print(f"✅ Created user DID: {did}")
    
    def test_full_verification_flow(self):
        """Test complete verification flow"""
        eth_address = "0x742d35Cc6634C0532925a3b844Bc9e7595f8c1F5"
        image_hash = hashlib.sha256(b"full flow test").hexdigest()
        
        # 1. Issue credential
        credential = self.service.issue_verification_credential(
            user_address=eth_address,
            image_hash=image_hash,
            is_real=True,
            confidence=0.95,
            oracle_signature="0x" + "a" * 130
        )
        
        assert credential is not None
        print(f"✅ Step 1: Credential issued")
        
        # 2. Verify credential
        result = self.service.verify_credential(credential)
        
        assert result.is_valid == True
        print(f"✅ Step 2: Credential verified")
        
        # 3. Prepare for blockchain
        blockchain_data = self.service.prepare_credential_for_blockchain(credential)
        
        assert "credentialHash" in blockchain_data
        assert "issuer" in blockchain_data
        print(f"✅ Step 3: Prepared for blockchain")
        print(f"   - Credential Hash: {blockchain_data['credentialHash'][:20]}...")
    
    def test_statistics(self):
        """Test statistics gathering"""
        # Create some DIDs and credentials
        self.service.create_user_did("0x1111111111111111111111111111111111111111")
        self.service.issue_verification_credential(
            user_address="0x2222222222222222222222222222222222222222",
            image_hash="abc123",
            is_real=True,
            confidence=0.9,
            oracle_signature="0xsig"
        )
        
        stats = self.service.get_statistics()
        
        assert "oracle" in stats
        assert "dids" in stats
        assert "credentials" in stats
        print(f"✅ Statistics gathered:")
        print(f"   - Oracle DID: {stats['oracle']['did'][:40]}...")
        print(f"   - Credentials issued: {stats['credentials']['total_issued']}")


def run_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("DID SYSTEM - COMPREHENSIVE TESTS")
    print("="*60 + "\n")
    
    test_classes = [
        TestKeyManager,
        TestDIDManager,
        TestCredentialIssuer,
        TestCredentialVerifier,
        TestDIDService
    ]
    
    results = {"passed": 0, "failed": 0}
    
    for test_class in test_classes:
        print(f"\n{'='*40}")
        print(f"Testing: {test_class.__name__}")
        print(f"{'='*40}")
        
        instance = test_class()
        
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    if hasattr(instance, "setup_method"):
                        instance.setup_method()
                    
                    getattr(instance, method_name)()
                    results["passed"] += 1
                except Exception as e:
                    print(f"❌ {method_name}: {e}")
                    results["failed"] += 1
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {results['passed']} passed, {results['failed']} failed")
    print(f"{'='*60}\n")
    
    return results["failed"] == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
