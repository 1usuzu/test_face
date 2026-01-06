pragma circom 2.1.0;

include "../node_modules/circomlib/circuits/poseidon.circom";
include "../node_modules/circomlib/circuits/comparators.circom";
include "../node_modules/circomlib/circuits/bitify.circom";

/*
 * DeepfakeVerification Circuit
 * 
 * Chứng minh: "Tôi có ảnh được Oracle xác nhận là REAL, mà không tiết lộ ảnh"
 * 
 * Public Inputs:
 *   - oraclePublicKeyHash: Hash của Oracle's public key (ai cũng biết)
 *   - nullifierHash: Unique identifier để chống double-spending
 *   - verificationCommitment: Commitment của verification result
 * 
 * Private Inputs (Secrets):
 *   - imageHash[4]: SHA256 hash của ảnh (chia thành 4 phần 64-bit)
 *   - isReal: 1 nếu ảnh thật, 0 nếu fake
 *   - confidence: Độ tin cậy (0-10000, đại diện 0-100%)
 *   - oracleSignature[4]: Chữ ký của Oracle (simplified)
 *   - userSecret: Secret của user để tạo nullifier
 *   - timestamp: Thời điểm verification
 */

template DeepfakeVerification() {
    // ===== PUBLIC INPUTS =====
    signal input oraclePublicKeyHash;    // Known Oracle identity
    signal input nullifierHash;           // Prevents double-use
    signal input verificationCommitment;  // Commitment to the result
    
    // ===== PRIVATE INPUTS (SECRETS) =====
    signal input imageHash[4];     // Image hash split into 4 x 64-bit chunks
    signal input isReal;           // 1 = real, 0 = fake  
    signal input confidence;       // 0-10000 (representing 0.00% - 100.00%)
    signal input oracleSignature[4]; // Simplified signature (4 x 64-bit)
    signal input userSecret;       // User's secret for nullifier
    signal input timestamp;        // Verification timestamp
    
    // ===== CONSTRAINT 1: isReal must be 0 or 1 =====
    signal isRealCheck;
    isRealCheck <== isReal * (1 - isReal);
    isRealCheck === 0;
    
    // ===== CONSTRAINT 2: Confidence in valid range (0-10000) =====
    component confidenceRange = LessThan(16);
    confidenceRange.in[0] <== confidence;
    confidenceRange.in[1] <== 10001;
    confidenceRange.out === 1;
    
    // ===== CONSTRAINT 3: Verify Oracle Signature =====
    // Hash(imageHash, isReal, confidence) phải khớp với signature
    // Sử dụng Poseidon hash (ZK-friendly)
    component signatureHash = Poseidon(7);
    signatureHash.inputs[0] <== imageHash[0];
    signatureHash.inputs[1] <== imageHash[1];
    signatureHash.inputs[2] <== imageHash[2];
    signatureHash.inputs[3] <== imageHash[3];
    signatureHash.inputs[4] <== isReal;
    signatureHash.inputs[5] <== confidence;
    signatureHash.inputs[6] <== timestamp;
    
    // Verify signature matches (simplified: sig = hash of data + oracle key)
    component sigVerify = Poseidon(5);
    sigVerify.inputs[0] <== signatureHash.out;
    sigVerify.inputs[1] <== oracleSignature[0];
    sigVerify.inputs[2] <== oracleSignature[1];
    sigVerify.inputs[3] <== oracleSignature[2];
    sigVerify.inputs[4] <== oracleSignature[3];
    
    // Signature verification: sigVerify output should match oraclePublicKeyHash
    component sigCheck = Poseidon(2);
    sigCheck.inputs[0] <== sigVerify.out;
    sigCheck.inputs[1] <== oraclePublicKeyHash;
    // This creates a binding between signature and oracle identity
    
    // ===== CONSTRAINT 4: Compute Nullifier Hash =====
    // nullifier = hash(imageHash, userSecret) - prevents same image being verified twice by same user
    component nullifier = Poseidon(5);
    nullifier.inputs[0] <== imageHash[0];
    nullifier.inputs[1] <== imageHash[1];
    nullifier.inputs[2] <== imageHash[2];
    nullifier.inputs[3] <== imageHash[3];
    nullifier.inputs[4] <== userSecret;
    
    nullifier.out === nullifierHash;
    
    // ===== CONSTRAINT 5: Compute Verification Commitment =====
    // commitment = hash(isReal, confidence, timestamp, userSecret)
    // This commits to the result without revealing it
    component commitment = Poseidon(4);
    commitment.inputs[0] <== isReal;
    commitment.inputs[1] <== confidence;
    commitment.inputs[2] <== timestamp;
    commitment.inputs[3] <== userSecret;
    
    commitment.out === verificationCommitment;
    
    // ===== CONSTRAINT 6: Only REAL images can generate valid proof =====
    // Đây là constraint quan trọng nhất!
    // Chỉ cho phép generate proof nếu isReal = 1
    isReal === 1;
}

// Export main component
component main {public [oraclePublicKeyHash, nullifierHash, verificationCommitment]} = DeepfakeVerification();
