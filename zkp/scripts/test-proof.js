/**
 * Script: Test ZK Proof generation and verification
 */

const snarkjs = require('snarkjs');
const fs = require('fs');
const path = require('path');

const CIRCUIT_NAME = 'simple_proof';
const BUILD_DIR = path.join(__dirname, '..', 'build');
const KEYS_DIR = path.join(__dirname, '..', 'keys');

// Poseidon hash implementation (simplified for testing)
// Trong thực tế, dùng library circomlib
const { buildPoseidon } = require('circomlibjs');

async function testProof() {
    console.log('🧪 Testing ZK Proof Generation & Verification\n');
    
    const wasmPath = path.join(BUILD_DIR, `${CIRCUIT_NAME}_js`, `${CIRCUIT_NAME}.wasm`);
    const zkeyPath = path.join(KEYS_DIR, `${CIRCUIT_NAME}.zkey`);
    const vkeyPath = path.join(KEYS_DIR, `${CIRCUIT_NAME}_verification_key.json`);
    
    // Check files exist
    if (!fs.existsSync(wasmPath) || !fs.existsSync(zkeyPath)) {
        console.error('❌ Missing files. Run compile and setup first!');
        console.log('   npm run compile');
        console.log('   npm run setup');
        process.exit(1);
    }
    
    try {
        // Initialize Poseidon
        const poseidon = await buildPoseidon();
        const F = poseidon.F;
        
        // ===== TEST CASE: Ảnh REAL =====
        console.log('📝 Test Case: User có ảnh REAL\n');
        
        // Giả lập dữ liệu từ Oracle
        const imageHash = BigInt('0x' + 'a'.repeat(64));  // Fake hash
        const isReal = 1n;  // Ảnh thật
        const oracleSecret = BigInt('0x' + 'b'.repeat(64));  // Oracle's secret
        const userSecret = BigInt('0x' + 'c'.repeat(32));  // User's secret
        
        // Tính nullifier (giống circuit)
        const nullifier = F.toObject(poseidon([imageHash, userSecret]));
        
        // Tính oracle check (giống circuit)
        const oracleCheck = F.toObject(poseidon([imageHash, isReal, oracleSecret]));
        
        // Tính commitment (giống circuit)
        const commitment = F.toObject(poseidon([oracleCheck, userSecret, nullifier]));
        
        console.log('Private Inputs (User giữ bí mật):');
        console.log(`  imageHash: ${imageHash.toString().substring(0, 20)}...`);
        console.log(`  isReal: ${isReal}`);
        console.log(`  userSecret: ${userSecret.toString().substring(0, 20)}...`);
        
        console.log('\nPublic Inputs (Blockchain thấy):');
        console.log(`  commitment: ${commitment.toString().substring(0, 30)}...`);
        console.log(`  nullifier: ${nullifier.toString().substring(0, 30)}...`);
        
        // Prepare circuit inputs
        const input = {
            // Public
            commitment: commitment.toString(),
            nullifier: nullifier.toString(),
            // Private
            imageHash: imageHash.toString(),
            isReal: isReal.toString(),
            oracleSecret: oracleSecret.toString(),
            userSecret: userSecret.toString()
        };
        
        console.log('\n⏳ Generating proof...');
        const startTime = Date.now();
        
        // Generate proof
        const { proof, publicSignals } = await snarkjs.groth16.fullProve(
            input,
            wasmPath,
            zkeyPath
        );
        
        const proofTime = Date.now() - startTime;
        console.log(`✅ Proof generated in ${proofTime}ms`);
        
        console.log('\n📦 Proof data:');
        console.log(`  pi_a: [${proof.pi_a[0].substring(0, 20)}..., ...]`);
        console.log(`  pi_b: [[...], [...]]`);
        console.log(`  pi_c: [${proof.pi_c[0].substring(0, 20)}..., ...]`);
        console.log(`  Public signals: ${publicSignals.length} values`);
        
        // Verify proof
        console.log('\n⏳ Verifying proof...');
        const vkey = JSON.parse(fs.readFileSync(vkeyPath));
        
        const verifyStart = Date.now();
        const isValid = await snarkjs.groth16.verify(vkey, publicSignals, proof);
        const verifyTime = Date.now() - verifyStart;
        
        if (isValid) {
            console.log(`✅ Proof VALID! Verified in ${verifyTime}ms`);
        } else {
            console.log('❌ Proof INVALID!');
        }
        
        // Export for Solidity
        console.log('\n📤 Exporting calldata for Solidity...');
        const calldata = await snarkjs.groth16.exportSolidityCallData(proof, publicSignals);
        console.log('Calldata (first 100 chars):', calldata.substring(0, 100) + '...');
        
        // Save test output
        const outputPath = path.join(__dirname, '..', 'test-output.json');
        fs.writeFileSync(outputPath, JSON.stringify({
            proof,
            publicSignals,
            calldata,
            isValid,
            timing: { proofTime, verifyTime }
        }, null, 2));
        console.log(`\n💾 Test output saved to ${outputPath}`);
        
    } catch (error) {
        console.error('❌ Test failed:', error);
        
        if (error.message.includes('Assert Failed')) {
            console.log('\n💡 This likely means the circuit constraints failed.');
            console.log('   Check if isReal === 1 (only REAL images can generate proofs)');
        }
        
        process.exit(1);
    }
}

testProof();
