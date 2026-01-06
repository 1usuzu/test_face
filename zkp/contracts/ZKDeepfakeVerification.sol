// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title IGroth16Verifier
 * @notice Interface cho Groth16 Verifier (generated bởi snarkjs)
 */
interface IGroth16Verifier {
    function verifyProof(
        uint[2] calldata _pA,
        uint[2][2] calldata _pB,
        uint[2] calldata _pC,
        uint[2] calldata _pubSignals  // [commitment, nullifier]
    ) external view returns (bool);
}

/**
 * @title ZKDeepfakeVerification
 * @notice Smart contract xác thực ZK Proof cho Deepfake Detection
 * 
 * Flow:
 * 1. User upload ảnh → AI verify → Oracle ký
 * 2. User generate ZK Proof (off-chain)
 * 3. User submit proof → Contract verify → Lưu commitment
 * 
 * Privacy: Contract KHÔNG biết:
 * - Ảnh gì (image hash)
 * - Kết quả cụ thể (isReal, confidence)
 * - Chỉ biết: "User đã verify thành công một ảnh REAL"
 */
contract ZKDeepfakeVerification {
    
    // ===== STATE =====
    IGroth16Verifier public verifier;
    address public owner;
    
    // Mapping để track nullifiers (chống double-spending)
    mapping(uint256 => bool) public usedNullifiers;
    
    // Mapping user => commitments (lịch sử verification)
    mapping(address => uint256[]) public userCommitments;
    
    // Struct lưu verification record
    struct VerificationRecord {
        address user;
        uint256 commitment;
        uint256 nullifier;
        uint256 timestamp;
        bool isValid;
    }
    
    // All verification records
    mapping(uint256 => VerificationRecord) public records;  // commitment => record
    uint256 public totalVerifications;
    
    // ===== EVENTS =====
    event ProofVerified(
        address indexed user,
        uint256 indexed commitment,
        uint256 nullifier,
        uint256 timestamp
    );
    
    event VerifierUpdated(address oldVerifier, address newVerifier);
    
    // ===== MODIFIERS =====
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }
    
    // ===== CONSTRUCTOR =====
    constructor(address _verifier) {
        owner = msg.sender;
        verifier = IGroth16Verifier(_verifier);
    }
    
    /**
     * @notice Verify ZK Proof và lưu kết quả
     * @param _pA Proof component A
     * @param _pB Proof component B  
     * @param _pC Proof component C
     * @param _pubSignals Public signals [commitment, nullifier]
     */
    function verifyAndRecord(
        uint[2] calldata _pA,
        uint[2][2] calldata _pB,
        uint[2] calldata _pC,
        uint[2] calldata _pubSignals
    ) external returns (bool) {
        uint256 commitment = _pubSignals[0];
        uint256 nullifier = _pubSignals[1];
        
        // 1. Check nullifier chưa được dùng
        require(!usedNullifiers[nullifier], "Nullifier already used");
        
        // 2. Verify ZK Proof
        bool isValid = verifier.verifyProof(_pA, _pB, _pC, _pubSignals);
        require(isValid, "Invalid ZK proof");
        
        // 3. Mark nullifier as used
        usedNullifiers[nullifier] = true;
        
        // 4. Store record
        records[commitment] = VerificationRecord({
            user: msg.sender,
            commitment: commitment,
            nullifier: nullifier,
            timestamp: block.timestamp,
            isValid: true
        });
        
        userCommitments[msg.sender].push(commitment);
        totalVerifications++;
        
        // 5. Emit event
        emit ProofVerified(msg.sender, commitment, nullifier, block.timestamp);
        
        return true;
    }
    
    /**
     * @notice Kiểm tra một commitment đã được verify chưa
     */
    function isCommitmentVerified(uint256 _commitment) external view returns (bool) {
        return records[_commitment].isValid;
    }
    
    /**
     * @notice Lấy lịch sử verification của user
     */
    function getUserCommitments(address _user) external view returns (uint256[] memory) {
        return userCommitments[_user];
    }
    
    /**
     * @notice Lấy record theo commitment
     */
    function getRecord(uint256 _commitment) external view returns (VerificationRecord memory) {
        return records[_commitment];
    }
    
    /**
     * @notice Update verifier contract (chỉ owner)
     */
    function updateVerifier(address _newVerifier) external onlyOwner {
        emit VerifierUpdated(address(verifier), _newVerifier);
        verifier = IGroth16Verifier(_newVerifier);
    }
    
    /**
     * @notice Batch verify multiple proofs (tiết kiệm gas)
     */
    function batchVerify(
        uint[2][] calldata _pAs,
        uint[2][2][] calldata _pBs,
        uint[2][] calldata _pCs,
        uint[2][] calldata _pubSignalsList
    ) external returns (uint256 successCount) {
        require(
            _pAs.length == _pBs.length && 
            _pBs.length == _pCs.length && 
            _pCs.length == _pubSignalsList.length,
            "Array length mismatch"
        );
        
        for (uint i = 0; i < _pAs.length; i++) {
            uint256 nullifier = _pubSignalsList[i][1];
            
            if (usedNullifiers[nullifier]) continue;
            
            bool isValid = verifier.verifyProof(
                _pAs[i], 
                _pBs[i], 
                _pCs[i], 
                _pubSignalsList[i]
            );
            
            if (isValid) {
                uint256 commitment = _pubSignalsList[i][0];
                usedNullifiers[nullifier] = true;
                
                records[commitment] = VerificationRecord({
                    user: msg.sender,
                    commitment: commitment,
                    nullifier: nullifier,
                    timestamp: block.timestamp,
                    isValid: true
                });
                
                userCommitments[msg.sender].push(commitment);
                totalVerifications++;
                successCount++;
                
                emit ProofVerified(msg.sender, commitment, nullifier, block.timestamp);
            }
        }
    }
}
