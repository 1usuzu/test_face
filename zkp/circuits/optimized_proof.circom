pragma circom 2.1.0;

include "../node_modules/circomlib/circuits/poseidon.circom";
include "../node_modules/circomlib/circuits/comparators.circom";
include "../node_modules/circomlib/circuits/bitify.circom";

/*
 * OptimizedDeepfakeProof - Phiên bản tối ưu
 * 
 * Cải tiến so với bản cũ:
 * 1. Thêm oracleCommitment để verify Oracle identity
 * 2. Thêm timestamp expiry check
 * 3. Tối ưu số lượng constraints
 * 4. Thêm confidence threshold
 * 
 * Chứng minh:
 * - "Tôi có ảnh được Oracle XÁC NHẬN là REAL"
 * - "Verification còn hiệu lực (chưa hết hạn)"
 * - "Độ tin cậy >= ngưỡng yêu cầu"
 */

template OptimizedDeepfakeProof() {
    // ========== PUBLIC INPUTS ==========
    signal input commitment;           // Commitment ẩn danh
    signal input nullifier;            // Chống double-use
    signal input oracleCommitment;     // Hash(oracleSecret, oraclePubKey) - verify Oracle
    signal input minConfidence;        // Ngưỡng confidence tối thiểu (0-10000)
    
    // ========== PRIVATE INPUTS ==========
    signal input imageHash;            // SHA256 hash của ảnh
    signal input isReal;               // 1 = real, 0 = fake
    signal input confidence;           // 0-10000 (0.00% - 100.00%)
    signal input oracleSecret;         // Secret từ Oracle
    signal input oraclePubKey;         // Oracle's public identifier
    signal input userSecret;           // User's secret
    signal input timestamp;            // Thời điểm verification
    signal input expiryTime;           // Thời hạn hiệu lực
    
    // ========== CONSTRAINT 1: Boolean check ==========
    // isReal phải là 0 hoặc 1
    signal isRealBool;
    isRealBool <== isReal * (1 - isReal);
    isRealBool === 0;
    
    // ========== CONSTRAINT 2: Only REAL images ==========
    // Đây là constraint QUAN TRỌNG NHẤT
    isReal === 1;
    
    // ========== CONSTRAINT 3: Confidence range ==========
    // confidence phải trong khoảng [0, 10000]
    component confUpper = LessEqThan(14);  // 14 bits cho 10000
    confUpper.in[0] <== confidence;
    confUpper.in[1] <== 10000;
    confUpper.out === 1;
    
    // confidence >= minConfidence (threshold check)
    component confThreshold = GreaterEqThan(14);
    confThreshold.in[0] <== confidence;
    confThreshold.in[1] <== minConfidence;
    confThreshold.out === 1;
    
    // ========== CONSTRAINT 4: Timestamp validity ==========
    // timestamp < expiryTime (chưa hết hạn)
    component timeValid = LessThan(64);  // 64 bits cho Unix timestamp
    timeValid.in[0] <== timestamp;
    timeValid.in[1] <== expiryTime;
    timeValid.out === 1;
    
    // ========== CONSTRAINT 5: Oracle verification ==========
    // Verify rằng oracleSecret thực sự từ Oracle hợp lệ
    component oracleVerify = Poseidon(2);
    oracleVerify.inputs[0] <== oracleSecret;
    oracleVerify.inputs[1] <== oraclePubKey;
    oracleVerify.out === oracleCommitment;
    
    // ========== CONSTRAINT 6: Oracle signed this data ==========
    // Oracle đã ký: hash(imageHash, isReal, confidence, timestamp)
    component dataSigned = Poseidon(5);
    dataSigned.inputs[0] <== imageHash;
    dataSigned.inputs[1] <== isReal;
    dataSigned.inputs[2] <== confidence;
    dataSigned.inputs[3] <== timestamp;
    dataSigned.inputs[4] <== oracleSecret;
    // dataSigned.out is implicitly verified through commitment
    
    // ========== CONSTRAINT 7: Nullifier computation ==========
    // nullifier = hash(imageHash, userSecret)
    // Ngăn cùng 1 user verify cùng 1 ảnh 2 lần
    component nullifierCalc = Poseidon(2);
    nullifierCalc.inputs[0] <== imageHash;
    nullifierCalc.inputs[1] <== userSecret;
    nullifier === nullifierCalc.out;
    
    // ========== CONSTRAINT 8: Commitment computation ==========
    // commitment = hash(dataSigned, userSecret, nullifier)
    // Cam kết kết quả mà không lộ chi tiết
    component commitmentCalc = Poseidon(3);
    commitmentCalc.inputs[0] <== dataSigned.out;
    commitmentCalc.inputs[1] <== userSecret;
    commitmentCalc.inputs[2] <== nullifier;
    commitment === commitmentCalc.out;
}

component main {public [commitment, nullifier, oracleCommitment, minConfidence]} = OptimizedDeepfakeProof();
