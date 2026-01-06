"""
DID Manager - Tạo và quản lý DID Documents theo chuẩn W3C DID Core 1.0

DID Format: did:deepfake:<unique-identifier>

Reference: https://www.w3.org/TR/did-core/
"""

import json
import hashlib
import secrets
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

from .key_manager import KeyManager, KeyPair


class DIDMethod(Enum):
    """Supported DID methods"""
    DEEPFAKE = "deepfake"  # Our custom method
    KEY = "key"            # did:key method
    ETH = "ethr"           # Ethereum DID


@dataclass
class ServiceEndpoint:
    """Service endpoint in DID Document"""
    id: str
    type: str
    service_endpoint: str
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "type": self.type,
            "serviceEndpoint": self.service_endpoint
        }
        if self.description:
            result["description"] = self.description
        return result


@dataclass
class DIDDocument:
    """
    W3C DID Document
    
    Reference: https://www.w3.org/TR/did-core/#core-properties
    """
    id: str  # The DID
    controller: Optional[str] = None
    verification_method: List[Dict] = field(default_factory=list)
    authentication: List[str] = field(default_factory=list)
    assertion_method: List[str] = field(default_factory=list)
    key_agreement: List[str] = field(default_factory=list)
    capability_invocation: List[str] = field(default_factory=list)
    capability_delegation: List[str] = field(default_factory=list)
    service: List[Dict] = field(default_factory=list)
    created: str = ""
    updated: str = ""
    deactivated: bool = False
    
    # Additional metadata
    also_known_as: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.controller:
            self.controller = self.id
        if not self.created:
            self.created = datetime.utcnow().isoformat() + "Z"
        if not self.updated:
            self.updated = self.created
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to W3C DID Document JSON format"""
        doc = {
            "@context": [
                "https://www.w3.org/ns/did/v1",
                "https://w3id.org/security/suites/ed25519-2020/v1",
                "https://w3id.org/security/suites/secp256k1-2019/v1"
            ],
            "id": self.id,
            "controller": self.controller,
            "verificationMethod": self.verification_method,
            "authentication": self.authentication,
            "assertionMethod": self.assertion_method,
        }
        
        # Optional fields
        if self.key_agreement:
            doc["keyAgreement"] = self.key_agreement
        if self.capability_invocation:
            doc["capabilityInvocation"] = self.capability_invocation
        if self.capability_delegation:
            doc["capabilityDelegation"] = self.capability_delegation
        if self.service:
            doc["service"] = self.service
        if self.also_known_as:
            doc["alsoKnownAs"] = self.also_known_as
            
        # Metadata
        doc["created"] = self.created
        doc["updated"] = self.updated
        
        if self.deactivated:
            doc["deactivated"] = self.deactivated
            
        return doc
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DIDDocument":
        """Create DIDDocument from dictionary"""
        return cls(
            id=data["id"],
            controller=data.get("controller"),
            verification_method=data.get("verificationMethod", []),
            authentication=data.get("authentication", []),
            assertion_method=data.get("assertionMethod", []),
            key_agreement=data.get("keyAgreement", []),
            capability_invocation=data.get("capabilityInvocation", []),
            capability_delegation=data.get("capabilityDelegation", []),
            service=data.get("service", []),
            created=data.get("created", ""),
            updated=data.get("updated", ""),
            deactivated=data.get("deactivated", False),
            also_known_as=data.get("alsoKnownAs", [])
        )


class DIDManager:
    """
    Manages DID creation, resolution, and updates
    
    Features:
    - Create new DIDs with cryptographic keys
    - Resolve DIDs to DID Documents
    - Update DID Documents
    - Deactivate DIDs
    - Link Ethereum address to DID
    """
    
    def __init__(self, key_manager: Optional[KeyManager] = None):
        self.key_manager = key_manager or KeyManager()
        self._documents: Dict[str, DIDDocument] = {}
        self._address_to_did: Dict[str, str] = {}  # Ethereum address -> DID
    
    # ==================== DID CREATION ====================
    
    def create_did(
        self,
        method: DIDMethod = DIDMethod.DEEPFAKE,
        ethereum_address: Optional[str] = None,
        include_eth_key: bool = True
    ) -> tuple[str, DIDDocument, Dict[str, KeyPair]]:
        """
        Create a new DID with associated keys
        
        Args:
            method: DID method to use
            ethereum_address: Optional Ethereum address to link
            include_eth_key: Whether to include secp256k1 key
            
        Returns:
            Tuple of (did, did_document, keys)
        """
        # Generate unique identifier
        unique_id = secrets.token_hex(16)
        
        if ethereum_address:
            # Use address hash for deterministic DID
            unique_id = hashlib.sha256(ethereum_address.lower().encode()).hexdigest()[:32]
        
        did = f"did:{method.value}:{unique_id}"
        
        # Generate keys
        keys = {}
        verification_methods = []
        authentication = []
        assertion_method = []
        
        # Primary Ed25519 key for signing
        ed_key = self.key_manager.generate_ed25519_keypair(did)
        keys[ed_key.key_id] = ed_key
        verification_methods.append(ed_key.to_verification_method())
        authentication.append(ed_key.key_id)
        assertion_method.append(ed_key.key_id)
        
        # Optional Ethereum key
        if include_eth_key:
            if ethereum_address:
                # Create key from existing address (without private key)
                eth_key = KeyPair(
                    key_id=f"{did}#key-eth-1",
                    key_type="EcdsaSecp256k1VerificationKey2019",
                    public_key=ethereum_address,
                    controller=did
                )
            else:
                eth_key = self.key_manager.generate_secp256k1_keypair(did)
            
            keys[eth_key.key_id] = eth_key
            verification_methods.append(eth_key.to_verification_method())
            authentication.append(eth_key.key_id)
        
        # Create DID Document
        did_doc = DIDDocument(
            id=did,
            verification_method=verification_methods,
            authentication=authentication,
            assertion_method=assertion_method
        )
        
        # Store
        self._documents[did] = did_doc
        
        if ethereum_address:
            self._address_to_did[ethereum_address.lower()] = did
        elif include_eth_key and keys.get(f"{did}#key-eth-1"):
            eth_address = keys[f"{did}#key-eth-1"].public_key
            self._address_to_did[eth_address.lower()] = did
        
        return did, did_doc, keys
    
    def create_did_from_ethereum(self, private_key: str) -> tuple[str, DIDDocument, Dict[str, KeyPair]]:
        """
        Create DID from existing Ethereum private key
        
        Args:
            private_key: Ethereum private key (hex with 0x prefix)
            
        Returns:
            Tuple of (did, did_document, keys)
        """
        from eth_account import Account
        account = Account.from_key(private_key)
        
        return self.create_did(
            method=DIDMethod.DEEPFAKE,
            ethereum_address=account.address,
            include_eth_key=True
        )
    
    # ==================== DID RESOLUTION ====================
    
    def resolve(self, did: str) -> Optional[DIDDocument]:
        """
        Resolve DID to DID Document
        
        Args:
            did: The DID to resolve
            
        Returns:
            DIDDocument if found and not deactivated, None otherwise
        """
        doc = self._documents.get(did)
        # Return None if deactivated
        if doc and doc.deactivated:
            return None
        return doc
    
    def resolve_by_address(self, address: str) -> Optional[DIDDocument]:
        """
        Resolve DID by Ethereum address
        
        Args:
            address: Ethereum address
            
        Returns:
            DIDDocument if found
        """
        did = self._address_to_did.get(address.lower())
        if did:
            return self._documents.get(did)
        return None
    
    def get_did_by_address(self, address: str) -> Optional[str]:
        """Get DID string by Ethereum address"""
        return self._address_to_did.get(address.lower())
    
    # ==================== DID UPDATES ====================
    
    def update_did(
        self, 
        did: str, 
        add_verification_methods: Optional[List[Dict]] = None,
        add_services: Optional[List[Dict]] = None,
        remove_verification_method_ids: Optional[List[str]] = None
    ) -> bool:
        """
        Update a DID Document
        
        Args:
            did: The DID to update
            add_verification_methods: List of verification methods to add
            add_services: List of services to add
            remove_verification_method_ids: IDs of verification methods to remove
            
        Returns:
            True if successful
        """
        doc = self._documents.get(did)
        if not doc:
            return False
        
        # Add verification methods
        if add_verification_methods:
            for vm in add_verification_methods:
                doc.verification_method.append(vm)
        
        # Add services
        if add_services:
            for svc in add_services:
                doc.service.append(svc)
        
        # Remove verification methods
        if remove_verification_method_ids:
            doc.verification_method = [
                vm for vm in doc.verification_method
                if vm.get("id") not in remove_verification_method_ids
            ]
        
        doc.updated = datetime.utcnow().isoformat() + "Z"
        return True
    
    def add_service(self, did: str, service: ServiceEndpoint) -> bool:
        """
        Add service endpoint to DID Document
        
        Args:
            did: The DID to update
            service: Service endpoint to add
            
        Returns:
            True if successful
        """
        doc = self._documents.get(did)
        if not doc:
            return False
        
        doc.service.append(service.to_dict())
        doc.updated = datetime.utcnow().isoformat() + "Z"
        return True
    
    def add_verification_method(self, did: str, key: KeyPair) -> bool:
        """Add new verification method to DID Document"""
        doc = self._documents.get(did)
        if not doc:
            return False
        
        doc.verification_method.append(key.to_verification_method())
        doc.updated = datetime.utcnow().isoformat() + "Z"
        return True
    
    def deactivate(self, did: str) -> bool:
        """Deactivate a DID"""
        doc = self._documents.get(did)
        if not doc:
            return False
        
        doc.deactivated = True
        doc.updated = datetime.utcnow().isoformat() + "Z"
        return True
    
    # ==================== UTILITIES ====================
    
    def export_document(self, did: str) -> Optional[str]:
        """Export DID Document as JSON"""
        doc = self._documents.get(did)
        if doc:
            return doc.to_json()
        return None
    
    def import_document(self, json_str: str) -> Optional[DIDDocument]:
        """Import DID Document from JSON"""
        try:
            data = json.loads(json_str)
            doc = DIDDocument.from_dict(data)
            self._documents[doc.id] = doc
            return doc
        except Exception as e:
            print(f"Failed to import DID Document: {e}")
            return None
    
    def list_dids(self) -> List[str]:
        """List all managed DIDs"""
        return list(self._documents.keys())
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about managed DIDs"""
        active = sum(1 for doc in self._documents.values() if not doc.deactivated)
        return {
            "total": len(self._documents),
            "active": active,
            "deactivated": len(self._documents) - active,
            "linked_addresses": len(self._address_to_did)
        }
