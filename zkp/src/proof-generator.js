/**
 * ZK Proof Generator - JavaScript Module
 * 
 * Module này dùng để generate ZK Proof từ kết quả AI
 * Có thể chạy trong Browser hoặc Node.js
 */

const snarkjs = require('snarkjs');
const { buildPoseidon } = require('circomlibjs');

class ZKProofGenerator {
    constructor(wasmPath, zkeyPath) {
        this.wasmPath = wasmPath;
        this.zkeyPath = zkeyPath;
        this.poseidon = null;
        this.F = null;
    }
    
    /**
     * Initialize Poseidon hash
     */
    async init() {
        this.poseidon = await buildPoseidon();
        this.F = this.poseidon.F;
        console.log('ZK Proof Generator initialized');
    }
    
    /**
     * Hash function wrapper
     */
    hash(inputs) {
        return this.F.toObject(this.poseidon(inputs));
    }
    
    /**
     * Generate ZK Proof từ kết quả AI verification
     * 
     * @param {Object} verificationResult - Kết quả từ AI API
     * @param {string} verificationResult.image_hash - SHA256 hash của ảnh
     * @param {string} verificationResult.label - "REAL" hoặc "FAKE"
     * @param {string} verificationResult.signature - Oracle signature
     * @param {string} userSecret - Secret của user (nên lưu an toàn)
     * @param {string} oracleSecret - Phần secret từ Oracle (trong signature)
     * 
     * @returns {Object} { proof, publicSignals, commitment, nullifier }
     */
    async generateProof(verificationResult, userSecret, oracleSecret) {
        if (!this.poseidon) {
            await this.init();
        }
        
        // Parse inputs
        const imageHash = BigInt('0x' + verificationResult.image_hash);
        const isReal = verificationResult.label === 'REAL' ? 1n : 0n;
        const userSecretBigInt = BigInt(userSecret);
        const oracleSecretBigInt = BigInt(oracleSecret);
        
        // Chỉ ảnh REAL mới generate được proof
        if (isReal !== 1n) {
            throw new Error('Cannot generate proof for FAKE images');
        }
        
        // Tính các giá trị cần thiết
        const nullifier = this.hash([imageHash, userSecretBigInt]);
        const oracleCheck = this.hash([imageHash, isReal, oracleSecretBigInt]);
        const commitment = this.hash([oracleCheck, userSecretBigInt, nullifier]);
        
        // Prepare circuit input
        const input = {
            // Public signals
            commitment: commitment.toString(),
            nullifier: nullifier.toString(),
            // Private signals
            imageHash: imageHash.toString(),
            isReal: isReal.toString(),
            oracleSecret: oracleSecretBigInt.toString(),
            userSecret: userSecretBigInt.toString()
        };
        
        console.log('Generating ZK proof...');
        const startTime = Date.now();
        
        // Generate proof
        const { proof, publicSignals } = await snarkjs.groth16.fullProve(
            input,
            this.wasmPath,
            this.zkeyPath
        );
        
        console.log(`Proof generated in ${Date.now() - startTime}ms`);
        
        return {
            proof,
            publicSignals,
            commitment: commitment.toString(),
            nullifier: nullifier.toString()
        };
    }
    
    /**
     * Export proof to Solidity calldata format
     */
    async toSolidityCalldata(proof, publicSignals) {
        const calldata = await snarkjs.groth16.exportSolidityCallData(proof, publicSignals);
        
        // Parse calldata
        const parts = calldata.split(',');
        
        return {
            pA: [parts[0].trim(), parts[1].trim()],
            pB: [[parts[2].trim(), parts[3].trim()], [parts[4].trim(), parts[5].trim()]],
            pC: [parts[6].trim(), parts[7].trim()],
            pubSignals: [parts[8].trim(), parts[9].trim()]
        };
    }
    
    /**
     * Verify proof locally (for testing)
     */
    async verifyProof(proof, publicSignals, vkeyPath) {
        const vkey = require(vkeyPath);
        return await snarkjs.groth16.verify(vkey, publicSignals, proof);
    }
}

// Export for Node.js
if (typeof module !== 'undefined') {
    module.exports = { ZKProofGenerator };
}
