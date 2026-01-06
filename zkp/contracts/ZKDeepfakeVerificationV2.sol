// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title IGroth16Verifier
 * @notice Interface for the Groth16 Verifier contract
 */
interface IGroth16Verifier {
    function verifyProof(
        uint[2] calldata _pA,
        uint[2][2] calldata _pB,
        uint[2] calldata _pC,
        uint[2] calldata _pubSignals
    ) external view returns (bool);
}

/**
 * @title ZKDeepfakeVerificationV2
 * @notice Optimized ZK Proof verification for Deepfake Detection
 * 
 * Improvements over V1:
 * - ReentrancyGuard protection
 * - Pausable for emergency
 * - Expiry time for verifications
 * - Gas optimizations
 * - Better event logging
 */
contract ZKDeepfakeVerificationV2 {
    
    // ===== STATE VARIABLES =====
    IGroth16Verifier public immutable verifier;  // Immutable for gas savings
    address public owner;
    bool public paused;
    
    // Verification expiry (default 30 days)
    uint256 public verificationValidityPeriod = 30 days;
    
    // Reentrancy guard
    uint256 private constant NOT_ENTERED = 1;
    uint256 private constant ENTERED = 2;
    uint256 private _status;
    
    // Nullifier tracking (compact storage)
    mapping(uint256 => bool) public usedNullifiers;
    
    // Verification records
    struct VerificationRecord {
        address user;
        uint256 commitment;
        uint256 nullifier;
        uint64 timestamp;      // Packed: 8 bytes
        uint64 expiresAt;      // Packed: 8 bytes
        bool isValid;
        bool isRevoked;
    }
    
    mapping(uint256 => VerificationRecord) public records;
    mapping(address => uint256[]) private userCommitmentsList;
    
    // Statistics
    uint256 public totalVerifications;
    uint256 public totalRevoked;
    
    // ===== EVENTS =====
    event ProofVerified(
        address indexed user,
        uint256 indexed commitment,
        uint256 nullifier,
        uint256 timestamp,
        uint256 expiresAt
    );
    
    event VerificationRevoked(
        uint256 indexed commitment,
        address indexed revokedBy,
        string reason
    );
    
    event Paused(address account);
    event Unpaused(address account);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event ValidityPeriodUpdated(uint256 oldPeriod, uint256 newPeriod);
    
    // ===== ERRORS (Gas efficient) =====
    error NotOwner();
    error ContractPaused();
    error NullifierAlreadyUsed();
    error InvalidProof();
    error ReentrancyDetected();
    error InvalidAddress();
    error VerificationNotFound();
    error AlreadyRevoked();
    error NotAuthorized();
    
    // ===== MODIFIERS =====
    modifier onlyOwner() {
        if (msg.sender != owner) revert NotOwner();
        _;
    }
    
    modifier whenNotPaused() {
        if (paused) revert ContractPaused();
        _;
    }
    
    modifier nonReentrant() {
        if (_status == ENTERED) revert ReentrancyDetected();
        _status = ENTERED;
        _;
        _status = NOT_ENTERED;
    }
    
    // ===== CONSTRUCTOR =====
    constructor(address _verifier) {
        if (_verifier == address(0)) revert InvalidAddress();
        verifier = IGroth16Verifier(_verifier);
        owner = msg.sender;
        _status = NOT_ENTERED;
    }
    
    // ===== CORE FUNCTIONS =====
    
    /**
     * @notice Verify ZK proof and record result
     * @param _pA Proof element A
     * @param _pB Proof element B
     * @param _pC Proof element C
     * @param _pubSignals Public signals [commitment, nullifier]
     * @return success Whether verification succeeded
     */
    function verifyAndRecord(
        uint[2] calldata _pA,
        uint[2][2] calldata _pB,
        uint[2] calldata _pC,
        uint[2] calldata _pubSignals
    ) external whenNotPaused nonReentrant returns (bool success) {
        uint256 commitment = _pubSignals[0];
        uint256 nullifier = _pubSignals[1];
        
        // Check nullifier not used
        if (usedNullifiers[nullifier]) revert NullifierAlreadyUsed();
        
        // Verify ZK proof
        bool isValid = verifier.verifyProof(_pA, _pB, _pC, _pubSignals);
        if (!isValid) revert InvalidProof();
        
        // Mark nullifier used
        usedNullifiers[nullifier] = true;
        
        // Calculate expiry
        uint64 timestamp = uint64(block.timestamp);
        uint64 expiresAt = uint64(block.timestamp + verificationValidityPeriod);
        
        // Store record
        records[commitment] = VerificationRecord({
            user: msg.sender,
            commitment: commitment,
            nullifier: nullifier,
            timestamp: timestamp,
            expiresAt: expiresAt,
            isValid: true,
            isRevoked: false
        });
        
        userCommitmentsList[msg.sender].push(commitment);
        
        unchecked {
            ++totalVerifications;
        }
        
        emit ProofVerified(msg.sender, commitment, nullifier, timestamp, expiresAt);
        
        return true;
    }
    
    /**
     * @notice Check if a verification is currently valid
     * @param _commitment The commitment to check
     * @return isValid Whether the verification is valid and not expired
     */
    function isVerificationValid(uint256 _commitment) external view returns (bool isValid) {
        VerificationRecord storage record = records[_commitment];
        return record.isValid && 
               !record.isRevoked && 
               block.timestamp < record.expiresAt;
    }
    
    /**
     * @notice Get verification record
     */
    function getRecord(uint256 _commitment) external view returns (VerificationRecord memory) {
        return records[_commitment];
    }
    
    /**
     * @notice Get user's commitments
     */
    function getUserCommitments(address _user) external view returns (uint256[] memory) {
        return userCommitmentsList[_user];
    }
    
    /**
     * @notice Get user's commitment count
     */
    function getUserCommitmentCount(address _user) external view returns (uint256) {
        return userCommitmentsList[_user].length;
    }
    
    // ===== ADMIN FUNCTIONS =====
    
    /**
     * @notice Revoke a verification (admin only)
     */
    function revokeVerification(uint256 _commitment, string calldata _reason) external onlyOwner {
        VerificationRecord storage record = records[_commitment];
        if (!record.isValid) revert VerificationNotFound();
        if (record.isRevoked) revert AlreadyRevoked();
        
        record.isRevoked = true;
        unchecked {
            ++totalRevoked;
        }
        
        emit VerificationRevoked(_commitment, msg.sender, _reason);
    }
    
    /**
     * @notice User can revoke their own verification
     */
    function selfRevoke(uint256 _commitment) external {
        VerificationRecord storage record = records[_commitment];
        if (record.user != msg.sender) revert NotAuthorized();
        if (record.isRevoked) revert AlreadyRevoked();
        
        record.isRevoked = true;
        unchecked {
            ++totalRevoked;
        }
        
        emit VerificationRevoked(_commitment, msg.sender, "Self-revoked");
    }
    
    /**
     * @notice Update validity period
     */
    function setValidityPeriod(uint256 _newPeriod) external onlyOwner {
        emit ValidityPeriodUpdated(verificationValidityPeriod, _newPeriod);
        verificationValidityPeriod = _newPeriod;
    }
    
    /**
     * @notice Pause contract
     */
    function pause() external onlyOwner {
        paused = true;
        emit Paused(msg.sender);
    }
    
    /**
     * @notice Unpause contract
     */
    function unpause() external onlyOwner {
        paused = false;
        emit Unpaused(msg.sender);
    }
    
    /**
     * @notice Transfer ownership
     */
    function transferOwnership(address _newOwner) external onlyOwner {
        if (_newOwner == address(0)) revert InvalidAddress();
        emit OwnershipTransferred(owner, _newOwner);
        owner = _newOwner;
    }
    
    // ===== VIEW FUNCTIONS =====
    
    /**
     * @notice Get statistics
     */
    function getStats() external view returns (
        uint256 _totalVerifications,
        uint256 _totalRevoked,
        bool _isPaused,
        uint256 _validityPeriod
    ) {
        return (totalVerifications, totalRevoked, paused, verificationValidityPeriod);
    }
}
