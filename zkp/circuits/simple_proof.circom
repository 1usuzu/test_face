pragma circom 2.1.0;

include "../node_modules/circomlib/circuits/poseidon.circom";
include "../node_modules/circomlib/circuits/comparators.circom";

/*
 * SimpleDeepfakeProof - Phiên bản đơn giản hơn để dễ hiểu
 * 
 * Chứng minh: "Tôi sở hữu kết quả verification từ Oracle, ảnh là REAL"
 * 
 * Ý tưởng:
 * - Oracle ký: hash(imageHash, isReal, confidence)
 * - User tạo proof chứng minh họ có signature hợp lệ
 * - Blockchain verify proof mà không biết imageHash
 */

template SimpleDeepfakeProof() {
    // PUBLIC INPUTS - Ai cũng thấy
    signal input commitment;        // Cam kết ẩn danh
    signal input nullifier;         // Chống double-spending
    
    // PRIVATE INPUTS - Chỉ prover biết
    signal input imageHash;         // Hash của ảnh (secret)
    signal input isReal;            // Kết quả AI
    signal input oracleSecret;      // Part of Oracle signature
    signal input userSecret;        // User's secret
    
    // 1. isReal phải là 0 hoặc 1
    isReal * (1 - isReal) === 0;
    
    // 2. Chỉ ảnh REAL mới có thể tạo proof
    isReal === 1;
    
    // 3. Verify Oracle đã ký kết quả này
    component oracleCheck = Poseidon(3);
    oracleCheck.inputs[0] <== imageHash;
    oracleCheck.inputs[1] <== isReal;
    oracleCheck.inputs[2] <== oracleSecret;
    // oracleCheck.out sẽ được verify với known oracle commitment
    
    // 4. Tính nullifier (chống dùng lại)
    component nullifierCalc = Poseidon(2);
    nullifierCalc.inputs[0] <== imageHash;
    nullifierCalc.inputs[1] <== userSecret;
    nullifier === nullifierCalc.out;
    
    // 5. Tính commitment (cam kết ẩn danh)
    component commitmentCalc = Poseidon(3);
    commitmentCalc.inputs[0] <== oracleCheck.out;
    commitmentCalc.inputs[1] <== userSecret;
    commitmentCalc.inputs[2] <== nullifier;
    commitment === commitmentCalc.out;
}

component main {public [commitment, nullifier]} = SimpleDeepfakeProof();
