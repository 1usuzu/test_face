// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract DeepfakeVerification {
    
    struct DIDDocument {
        address owner;
        string did;
        string publicKeyBase58;
        bool isActive;
        uint256 createdAt;
        uint256 updatedAt;
    }
    
    struct VerificationResult {
        bytes32 imageHash;
        string subjectDid;
        string issuerDid;
        bool isReal;
        uint256 confidence;
        uint256 timestamp;
        bytes32 credentialHash;
    }
    
    mapping(address => DIDDocument) public didDocuments;
    mapping(string => address) public didToAddress;
    mapping(bytes32 => VerificationResult) public verificationResults;
    mapping(address => bool) public authorizedIssuers;
    
    address public owner;
    uint256 public totalDIDs;
    uint256 public totalVerifications;
    
    event DIDRegistered(address indexed owner, string did, uint256 timestamp);
    event VerificationRecorded(bytes32 indexed imageHash, string subjectDid, bool isReal, uint256 confidence, uint256 timestamp);
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }
    
    modifier didExists(address _owner) {
        require(didDocuments[_owner].isActive, "DID does not exist");
        _;
    }
    
    constructor() {
        owner = msg.sender;
        authorizedIssuers[msg.sender] = true; 
    }
    
    // --- HELPER FUNCTIONS CHO CHỮ KÝ ---
    function splitSignature(bytes memory sig) internal pure returns (bytes32 r, bytes32 s, uint8 v) {
        require(sig.length == 65, "Invalid signature length");
        assembly {
            r := mload(add(sig, 32))
            s := mload(add(sig, 64))
            v := byte(0, mload(add(sig, 96)))
        }
    }
    
    function toAsciiString(address x) internal pure returns (string memory) {
        bytes memory s = new bytes(42);
        s[0] = '0';
        s[1] = 'x';
        for (uint i = 0; i < 20; i++) {
            bytes1 b = bytes1(uint8(uint(uint160(x)) / (2**(8*(19 - i)))));
            bytes1 hi = bytes1(uint8(b) / 16);
            bytes1 lo = bytes1(uint8(b) - 16 * uint8(hi));
            s[2+i*2] = char(hi);
            s[3+i*2] = char(lo);
        }
        return string(s);
    }
    
    function char(bytes1 b) internal pure returns (bytes1 c) {
        if (uint8(b) < 10) return bytes1(uint8(b) + 0x30);
        else return bytes1(uint8(b) + 0x57);
    }
    
    function bytes32ToHexString(bytes32 _bytes) internal pure returns (string memory) {
        bytes memory s = new bytes(64);
        for (uint i = 0; i < 32; i++) {
            bytes1 b = _bytes[i];
            bytes1 hi = bytes1(uint8(b) / 16);
            bytes1 lo = bytes1(uint8(b) - 16 * uint8(hi));
            s[i*2] = char(hi);
            s[i*2+1] = char(lo);
        }
        return string(s);
    }
    
    function uint2str(uint256 _i) internal pure returns (string memory) {
        if (_i == 0) return "0";
        uint256 j = _i;
        uint256 length;
        while (j != 0) { length++; j /= 10; }
        bytes memory bstr = new bytes(length);
        uint256 k = length;
        while (_i != 0) {
            k = k-1;
            uint8 temp = uint8(48 + _i % 10);
            bstr[k] = bytes1(temp);
            _i /= 10;
        }
        return string(bstr);
    }

    function recoverSigner(bytes32 _ethSignedMessageHash, bytes memory _signature) internal pure returns (address) {
        (bytes32 r, bytes32 s, uint8 v) = splitSignature(_signature);
        return ecrecover(_ethSignedMessageHash, v, r, s);
    }

    // --- LOGIC CHÍNH: XÁC THỰC VÀ GHI ---
    function recordVerification(
        bytes32 _imageHash,
        bool _isReal,
        uint256 _confidence,
        bytes calldata _signature  // <--- Nhận chữ ký từ Frontend
    ) external didExists(msg.sender) {
        
        // 1. Tái tạo Message Hash (Phải khớp 100% với cách Python backend tạo chuỗi)
        // Format: "UserAddress:ImageHash:IsReal"
        string memory isRealStr = _isReal ? "true" : "false";
        
        // Convert bytes32 to hex string to match backend format
        string memory hashHex = bytes32ToHexString(_imageHash);
        
        // Hash nội dung thô - format: "0xAddress:hexHash:true/false"
        bytes32 messageHash = keccak256(abi.encodePacked(
            toAsciiString(msg.sender), ":", hashHex, ":", isRealStr
        ));
        
        // Thêm prefix Ethereum ("\x19Ethereum Signed Message:\n{length}")
        // Độ dài message = 42 (address) + 1 (:) + 64 (hash) + 1 (:) + 4/5 (true/false) = 112/113
        uint256 msgLen = 42 + 1 + 64 + 1 + bytes(isRealStr).length;
        bytes32 ethSignedMessageHash = keccak256(abi.encodePacked(
            "\x19Ethereum Signed Message:\n",
            uint2str(msgLen),
            toAsciiString(msg.sender), ":", hashHex, ":", isRealStr
        ));

        // 2. Tìm ra địa chỉ người đã ký
        address signer = recoverSigner(ethSignedMessageHash, _signature);

        // 3. KIỂM TRA: Người ký có phải là Server (Authorized Issuer) không?
        require(authorizedIssuers[signer], "Invalid signature! Data corrupted or fake.");

        // 4. Lưu kết quả
        verificationResults[_imageHash] = VerificationResult({
            imageHash: _imageHash,
            subjectDid: didDocuments[msg.sender].did,
            issuerDid: didDocuments[signer].did, // Issuer là Server
            isReal: _isReal,
            confidence: _confidence,
            timestamp: block.timestamp,
            credentialHash: keccak256(abi.encodePacked(_signature))
        });
        
        totalVerifications++;
        emit VerificationRecorded(_imageHash, didDocuments[msg.sender].did, _isReal, _confidence, block.timestamp);
    }
    
    // --- CÁC HÀM KHÁC (GIỮ NGUYÊN HOẶC RÚT GỌN) ---
    function registerDID(string calldata _did, string calldata _publicKeyBase58) external {
        require(!didDocuments[msg.sender].isActive, "DID already registered");
        
        didDocuments[msg.sender] = DIDDocument({
            owner: msg.sender,
            did: _did,
            publicKeyBase58: _publicKeyBase58,
            isActive: true,
            createdAt: block.timestamp,
            updatedAt: block.timestamp
        });
        didToAddress[_did] = msg.sender;
        totalDIDs++;
        emit DIDRegistered(msg.sender, _did, block.timestamp);
    }

    function getVerification(bytes32 _imageHash) external view returns (VerificationResult memory) {
        return verificationResults[_imageHash];
    }
    
    function authorizeIssuer(address _issuer) external onlyOwner {
        authorizedIssuers[_issuer] = true;
    }

    function getStats() external view returns (uint256, uint256) {
        return (totalDIDs, totalVerifications);
    }
}