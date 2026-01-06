/**
 * Browser-compatible ZK Proof Generator
 * 
 * Dùng cho Frontend React
 */

import * as snarkjs from 'snarkjs';

// Poseidon hash constants (pre-computed for browser)
// Trong thực tế, cần import từ circomlibjs hoặc compute on-demand

export class BrowserZKProof {
    constructor() {
        this.wasmPath = '/zkp/simple_proof.wasm';
        this.zkeyPath = '/zkp/simple_proof.zkey';
        this.initialized = false;
    }

    /**
     * Initialize - load WASM và zkey
     */
    async init() {
        // Preload files
        console.log('Loading ZK proving files...');
        this.initialized = true;
    }

    /**
     * Generate commitment từ verification result
     * Sử dụng Web Crypto API cho hashing
     */
    async computeCommitment(imageHash, isReal, userSecret) {
        // Simplified: dùng SHA256 thay vì Poseidon cho browser
        // Trong production, cần port Poseidon sang WASM
        const encoder = new TextEncoder();
        const data = encoder.encode(`${imageHash}:${isReal}:${userSecret}`);
        const hashBuffer = await crypto.subtle.digest('SHA-256', data);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        return '0x' + hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    }

    /**
     * Generate nullifier
     */
    async computeNullifier(imageHash, userSecret) {
        const encoder = new TextEncoder();
        const data = encoder.encode(`nullifier:${imageHash}:${userSecret}`);
        const hashBuffer = await crypto.subtle.digest('SHA-256', data);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        return '0x' + hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    }

    /**
     * Generate ZK Proof
     * 
     * @param {Object} verificationResult - Kết quả từ AI API
     * @param {string} userSecret - User's secret (stored locally)
     */
    async generateProof(verificationResult, userSecret) {
        if (verificationResult.label !== 'REAL') {
            throw new Error('Chỉ có thể tạo proof cho ảnh REAL');
        }

        const imageHash = verificationResult.image_hash;
        const signature = verificationResult.signature;

        // Compute values
        const nullifier = await this.computeNullifier(imageHash, userSecret);
        const commitment = await this.computeCommitment(imageHash, '1', userSecret);

        // Trong production, đây là nơi gọi snarkjs.groth16.fullProve
        // Cần WASM file và zkey file được serve từ server

        console.log('Generating ZK proof in browser...');

        try {
            // Circuit inputs
            const input = {
                commitment: BigInt(commitment).toString(),
                nullifier: BigInt(nullifier).toString(),
                imageHash: BigInt('0x' + imageHash).toString(),
                isReal: '1',
                oracleSecret: BigInt('0x' + signature.substring(0, 64)).toString(),
                userSecret: BigInt(userSecret).toString()
            };

            // Generate proof
            const { proof, publicSignals } = await snarkjs.groth16.fullProve(
                input,
                this.wasmPath,
                this.zkeyPath
            );

            // Format for Solidity
            const calldata = await snarkjs.groth16.exportSolidityCallData(proof, publicSignals);

            return {
                proof,
                publicSignals,
                calldata,
                commitment,
                nullifier,
                success: true
            };
        } catch (error) {
            console.error('ZK Proof generation failed:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Parse calldata for contract call
     */
    parseCalldata(calldata) {
        // calldata format: ["0x...", "0x..."],[[...],[...]],["0x...","0x..."],["0x...","0x..."]
        const regex = /\["(0x[^"]+)","(0x[^"]+)"\],\[\["(0x[^"]+)","(0x[^"]+)"\],\["(0x[^"]+)","(0x[^"]+)"\]\],\["(0x[^"]+)","(0x[^"]+)"\],\["(0x[^"]+)","(0x[^"]+)"\]/;
        const match = calldata.match(regex);
        
        if (!match) {
            throw new Error('Invalid calldata format');
        }

        return {
            pA: [match[1], match[2]],
            pB: [[match[3], match[4]], [match[5], match[6]]],
            pC: [match[7], match[8]],
            pubSignals: [match[9], match[10]]
        };
    }
}

export default BrowserZKProof;
