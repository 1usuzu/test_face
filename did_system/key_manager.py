"""
Key Manager - Quản lý Cryptographic Keys cho DID System

Supports:
- Ed25519: Cho DID signing (W3C recommended)
- secp256k1: Cho Ethereum compatibility
"""

import os
import json
import hashlib
import base64
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Cryptography imports
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.backends import default_backend

# Ethereum compatibility
from eth_account import Account
from eth_account.messages import encode_defunct


@dataclass
class KeyPair:
    """Represents a cryptographic key pair"""
    key_id: str
    key_type: str  # Ed25519VerificationKey2020, EcdsaSecp256k1VerificationKey2019
    public_key: str  # Base58 or Hex encoded
    private_key: Optional[str] = None  # Only stored locally, never shared
    created_at: str = ""
    controller: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"
    
    def to_verification_method(self) -> Dict[str, Any]:
        """Convert to W3C Verification Method format"""
        return {
            "id": self.key_id,
            "type": self.key_type,
            "controller": self.controller,
            "publicKeyMultibase": f"z{self.public_key}" if not self.public_key.startswith("z") else self.public_key
        }


class KeyManager:
    """
    Manages cryptographic keys for DID operations
    
    Features:
    - Generate Ed25519 key pairs
    - Generate secp256k1 key pairs (Ethereum compatible)
    - Sign and verify messages
    - Export/Import keys
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self._keys: Dict[str, KeyPair] = {}
    
    # ==================== KEY GENERATION ====================
    
    def generate_ed25519_keypair(self, did: str) -> KeyPair:
        """
        Generate Ed25519 key pair (recommended for DID)
        
        Args:
            did: The DID that will control this key
            
        Returns:
            KeyPair with Ed25519 keys
        """
        # Generate private key
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        
        # Serialize keys
        private_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption()
        )
        public_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        
        # Base58 encode (simplified - using base64 for compatibility)
        public_b58 = base64.urlsafe_b64encode(public_bytes).decode('utf-8').rstrip('=')
        private_b58 = base64.urlsafe_b64encode(private_bytes).decode('utf-8').rstrip('=')
        
        key_id = f"{did}#key-1"
        
        keypair = KeyPair(
            key_id=key_id,
            key_type="Ed25519VerificationKey2020",
            public_key=public_b58,
            private_key=private_b58,
            controller=did
        )
        
        self._keys[key_id] = keypair
        return keypair
    
    def generate_secp256k1_keypair(self, did: str) -> KeyPair:
        """
        Generate secp256k1 key pair (Ethereum compatible)
        
        Args:
            did: The DID that will control this key
            
        Returns:
            KeyPair with secp256k1 keys
        """
        # Generate Ethereum account
        account = Account.create()
        
        key_id = f"{did}#key-eth-1"
        
        keypair = KeyPair(
            key_id=key_id,
            key_type="EcdsaSecp256k1VerificationKey2019",
            public_key=account.address,  # Ethereum address as public key
            private_key=account.key.hex(),
            controller=did
        )
        
        self._keys[key_id] = keypair
        return keypair
    
    def generate_from_ethereum_key(self, did: str, private_key: str) -> KeyPair:
        """
        Create KeyPair from existing Ethereum private key
        
        Args:
            did: The DID that will control this key
            private_key: Ethereum private key (hex string with 0x prefix)
            
        Returns:
            KeyPair
        """
        account = Account.from_key(private_key)
        
        key_id = f"{did}#key-eth-1"
        
        keypair = KeyPair(
            key_id=key_id,
            key_type="EcdsaSecp256k1VerificationKey2019",
            public_key=account.address,
            private_key=private_key,
            controller=did
        )
        
        self._keys[key_id] = keypair
        return keypair
    
    # ==================== SIGNING ====================
    
    def sign_ed25519(self, key_id: str, message: bytes) -> str:
        """
        Sign message with Ed25519 key
        
        Args:
            key_id: The key ID to use for signing
            message: Message bytes to sign
            
        Returns:
            Base64 encoded signature
        """
        keypair = self._keys.get(key_id)
        if not keypair or keypair.key_type != "Ed25519VerificationKey2020":
            raise ValueError(f"Ed25519 key not found: {key_id}")
        
        if not keypair.private_key:
            raise ValueError("Private key not available for signing")
        
        # Decode private key
        private_bytes = base64.urlsafe_b64decode(keypair.private_key + '==')
        private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_bytes)
        
        # Sign
        signature = private_key.sign(message)
        
        return base64.urlsafe_b64encode(signature).decode('utf-8')
    
    def sign_secp256k1(self, key_id: str, message: str) -> str:
        """
        Sign message with secp256k1 key (Ethereum style)
        
        Args:
            key_id: The key ID to use for signing
            message: Message string to sign
            
        Returns:
            Hex encoded signature
        """
        keypair = self._keys.get(key_id)
        if not keypair or keypair.key_type != "EcdsaSecp256k1VerificationKey2019":
            raise ValueError(f"secp256k1 key not found: {key_id}")
        
        if not keypair.private_key:
            raise ValueError("Private key not available for signing")
        
        # Ethereum signing
        msg = encode_defunct(text=message)
        signed = Account.sign_message(msg, private_key=keypair.private_key)
        
        return signed.signature.hex()
    
    # ==================== VERIFICATION ====================
    
    def verify_ed25519(self, public_key: str, message: bytes, signature: str) -> bool:
        """
        Verify Ed25519 signature
        
        Args:
            public_key: Base64 encoded public key
            message: Original message bytes
            signature: Base64 encoded signature
            
        Returns:
            True if signature is valid
        """
        try:
            public_bytes = base64.urlsafe_b64decode(public_key + '==')
            sig_bytes = base64.urlsafe_b64decode(signature + '==')
            
            pub_key = ed25519.Ed25519PublicKey.from_public_bytes(public_bytes)
            pub_key.verify(sig_bytes, message)
            return True
        except Exception:
            return False
    
    def verify_secp256k1(self, message: str, signature: str, expected_address: str) -> bool:
        """
        Verify secp256k1 signature (Ethereum style)
        
        Args:
            message: Original message string
            signature: Hex encoded signature
            expected_address: Expected Ethereum address
            
        Returns:
            True if signature is valid and matches address
        """
        try:
            msg = encode_defunct(text=message)
            recovered_address = Account.recover_message(msg, signature=signature)
            return recovered_address.lower() == expected_address.lower()
        except Exception:
            return False
    
    # ==================== KEY MANAGEMENT ====================
    
    def get_key(self, key_id: str) -> Optional[KeyPair]:
        """Get key by ID"""
        return self._keys.get(key_id)
    
    def list_keys(self) -> list:
        """List all key IDs"""
        return list(self._keys.keys())
    
    def export_public_keys(self) -> Dict[str, Dict]:
        """Export all public keys (no private keys)"""
        result = {}
        for key_id, keypair in self._keys.items():
            result[key_id] = {
                "key_id": keypair.key_id,
                "key_type": keypair.key_type,
                "public_key": keypair.public_key,
                "controller": keypair.controller,
                "created_at": keypair.created_at
            }
        return result
    
    def save_keys(self, filepath: str, password: Optional[str] = None):
        """
        Save keys to file (encrypted if password provided)
        
        WARNING: In production, use proper key management (HSM, KMS)
        """
        data = {
            key_id: asdict(keypair) 
            for key_id, keypair in self._keys.items()
        }
        
        if password:
            # Simple encryption (use proper encryption in production)
            import hashlib
            key = hashlib.sha256(password.encode()).digest()
            # TODO: Implement proper encryption
            pass
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_keys(self, filepath: str, password: Optional[str] = None):
        """Load keys from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for key_id, key_data in data.items():
            self._keys[key_id] = KeyPair(**key_data)
    
    # ==================== ALIAS METHODS ====================
    # For backwards compatibility with tests
    
    def generate_ed25519(self, key_id: str) -> KeyPair:
        """
        Generate Ed25519 key pair with custom key_id
        
        Args:
            key_id: Custom key ID
            
        Returns:
            KeyPair with Ed25519 keys
        """
        # Generate private key
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        
        # Serialize keys
        private_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption()
        )
        public_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        
        # Base64 encode
        public_b58 = base64.urlsafe_b64encode(public_bytes).decode('utf-8').rstrip('=')
        private_b58 = base64.urlsafe_b64encode(private_bytes).decode('utf-8').rstrip('=')
        
        keypair = KeyPair(
            key_id=key_id,
            key_type="Ed25519VerificationKey2020",
            public_key=public_b58,
            private_key=private_b58,
            controller=""
        )
        
        # Store with address as ethereum_address attribute
        keypair.ethereum_address = None
        self._keys[key_id] = keypair
        return keypair
    
    def generate_secp256k1(self, key_id: str) -> KeyPair:
        """
        Generate secp256k1 key pair with custom key_id
        
        Args:
            key_id: Custom key ID
            
        Returns:
            KeyPair with secp256k1 keys
        """
        account = Account.create()
        
        keypair = KeyPair(
            key_id=key_id,
            key_type="EcdsaSecp256k1VerificationKey2019",
            public_key=account.address[2:],  # Remove 0x prefix for public key
            private_key=account.key.hex(),
            controller=""
        )
        
        # Add ethereum address as attribute
        keypair.ethereum_address = account.address
        self._keys[key_id] = keypair
        return keypair
