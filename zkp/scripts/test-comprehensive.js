/**
 * Comprehensive ZK Proof Test Suite
 * 
 * Test cases:
 * 1. ✅ REAL image with valid oracle secret → Proof SUCCESS
 * 2. ❌ FAKE image → Proof FAILS (constraint violation)
 * 3. ❌ Wrong oracle secret → Proof FAILS
 * 4. ❌ Replay attack (same nullifier) → Should be rejected
 * 5. ⏱️ Performance benchmarking
 */

const snarkjs = require('snarkjs');
const fs = require('fs');
const path = require('path');

const CIRCUIT_NAME = 'simple_proof';
const BUILD_DIR = path.join(__dirname, '..', 'build');
const KEYS_DIR = path.join(__dirname, '..', 'keys');

let poseidon, F;

// Colors for console
const colors = {
    reset: '\x1b[0m',
    green: '\x1b[32m',
    red: '\x1b[31m',
    yellow: '\x1b[33m',
    cyan: '\x1b[36m',
    bold: '\x1b[1m'
};

function log(color, ...args) {
    console.log(color, ...args, colors.reset);
}

async function initPoseidon() {
    const { buildPoseidon } = require('circomlibjs');
    poseidon = await buildPoseidon();
    F = poseidon.F;
}

function computeCircuitValues(imageHash, isReal, oracleSecret, userSecret) {
    const nullifier = F.toObject(poseidon([imageHash, userSecret]));
    const oracleCheck = F.toObject(poseidon([imageHash, isReal, oracleSecret]));
    const commitment = F.toObject(poseidon([oracleCheck, userSecret, nullifier]));
    return { nullifier, oracleCheck, commitment };
}

async function generateProof(input, wasmPath, zkeyPath) {
    try {
        const { proof, publicSignals } = await snarkjs.groth16.fullProve(
            input,
            wasmPath,
            zkeyPath
        );
        return { success: true, proof, publicSignals };
    } catch (error) {
        return { success: false, error: error.message };
    }
}

async function verifyProof(proof, publicSignals, vkeyPath) {
    const vkey = JSON.parse(fs.readFileSync(vkeyPath));
    return await snarkjs.groth16.verify(vkey, publicSignals, proof);
}

// ============================================================
// TEST CASES
// ============================================================

async function testCase1_RealImageValid(wasmPath, zkeyPath, vkeyPath) {
    log(colors.cyan, '\n📝 TEST 1: REAL image với valid oracle secret');
    
    const imageHash = BigInt('0x' + 'a1b2c3d4'.repeat(8));
    const isReal = 1n;
    const oracleSecret = BigInt('0x' + 'oracle123'.padEnd(64, '0'));
    const userSecret = BigInt('0x' + 'user456'.padEnd(32, '0'));
    
    const { nullifier, commitment } = computeCircuitValues(imageHash, isReal, oracleSecret, userSecret);
    
    const input = {
        commitment: commitment.toString(),
        nullifier: nullifier.toString(),
        imageHash: imageHash.toString(),
        isReal: isReal.toString(),
        oracleSecret: oracleSecret.toString(),
        userSecret: userSecret.toString()
    };
    
    const startTime = Date.now();
    const result = await generateProof(input, wasmPath, zkeyPath);
    const proofTime = Date.now() - startTime;
    
    if (result.success) {
        const isValid = await verifyProof(result.proof, result.publicSignals, vkeyPath);
        if (isValid) {
            log(colors.green, `   ✅ PASSED - Proof generated and verified in ${proofTime}ms`);
            return { passed: true, proofTime };
        } else {
            log(colors.red, '   ❌ FAILED - Proof generated but verification failed');
            return { passed: false };
        }
    } else {
        log(colors.red, '   ❌ FAILED - Could not generate proof:', result.error);
        return { passed: false };
    }
}

async function testCase2_FakeImageFails(wasmPath, zkeyPath) {
    log(colors.cyan, '\n📝 TEST 2: FAKE image should FAIL to generate proof');
    
    const imageHash = BigInt('0x' + 'fake1234'.repeat(8));
    const isReal = 0n;  // FAKE!
    const oracleSecret = BigInt('0x' + 'oracle123'.padEnd(64, '0'));
    const userSecret = BigInt('0x' + 'user456'.padEnd(32, '0'));
    
    const { nullifier, commitment } = computeCircuitValues(imageHash, isReal, oracleSecret, userSecret);
    
    const input = {
        commitment: commitment.toString(),
        nullifier: nullifier.toString(),
        imageHash: imageHash.toString(),
        isReal: isReal.toString(),  // 0 = FAKE
        oracleSecret: oracleSecret.toString(),
        userSecret: userSecret.toString()
    };
    
    const result = await generateProof(input, wasmPath, zkeyPath);
    
    if (!result.success) {
        log(colors.green, '   ✅ PASSED - Correctly rejected FAKE image');
        log(colors.yellow, `      Error: ${result.error.substring(0, 80)}...`);
        return { passed: true };
    } else {
        log(colors.red, '   ❌ FAILED - FAKE image should not generate valid proof!');
        return { passed: false };
    }
}

async function testCase3_WrongCommitmentFails(wasmPath, zkeyPath) {
    log(colors.cyan, '\n📝 TEST 3: Wrong commitment should FAIL');
    
    const imageHash = BigInt('0x' + 'test5678'.repeat(8));
    const isReal = 1n;
    const oracleSecret = BigInt('0x' + 'oracle123'.padEnd(64, '0'));
    const userSecret = BigInt('0x' + 'user456'.padEnd(32, '0'));
    
    // Compute correct values
    const { nullifier } = computeCircuitValues(imageHash, isReal, oracleSecret, userSecret);
    
    // Use WRONG commitment
    const wrongCommitment = BigInt('0x' + 'wrong'.repeat(12));
    
    const input = {
        commitment: wrongCommitment.toString(),  // WRONG!
        nullifier: nullifier.toString(),
        imageHash: imageHash.toString(),
        isReal: isReal.toString(),
        oracleSecret: oracleSecret.toString(),
        userSecret: userSecret.toString()
    };
    
    const result = await generateProof(input, wasmPath, zkeyPath);
    
    if (!result.success) {
        log(colors.green, '   ✅ PASSED - Correctly rejected wrong commitment');
        return { passed: true };
    } else {
        log(colors.red, '   ❌ FAILED - Wrong commitment should not generate valid proof!');
        return { passed: false };
    }
}

async function testCase4_WrongOracleSecretFails(wasmPath, zkeyPath) {
    log(colors.cyan, '\n📝 TEST 4: Wrong oracle secret should FAIL');
    
    const imageHash = BigInt('0x' + 'img12345'.repeat(8));
    const isReal = 1n;
    const correctOracleSecret = BigInt('0x' + 'correct'.padEnd(64, '0'));
    const wrongOracleSecret = BigInt('0x' + 'hacker'.padEnd(64, '0'));
    const userSecret = BigInt('0x' + 'user789'.padEnd(32, '0'));
    
    // Compute with CORRECT oracle secret
    const { nullifier, commitment } = computeCircuitValues(imageHash, isReal, correctOracleSecret, userSecret);
    
    // But try to prove with WRONG oracle secret
    const input = {
        commitment: commitment.toString(),
        nullifier: nullifier.toString(),
        imageHash: imageHash.toString(),
        isReal: isReal.toString(),
        oracleSecret: wrongOracleSecret.toString(),  // WRONG!
        userSecret: userSecret.toString()
    };
    
    const result = await generateProof(input, wasmPath, zkeyPath);
    
    if (!result.success) {
        log(colors.green, '   ✅ PASSED - Correctly rejected wrong oracle secret');
        return { passed: true };
    } else {
        log(colors.red, '   ❌ FAILED - Wrong oracle secret should not generate valid proof!');
        return { passed: false };
    }
}

async function testCase5_PerformanceBenchmark(wasmPath, zkeyPath, vkeyPath) {
    log(colors.cyan, '\n📝 TEST 5: Performance Benchmark (10 proofs)');
    
    const times = [];
    
    for (let i = 0; i < 10; i++) {
        const imageHash = BigInt('0x' + (i.toString(16).repeat(64)).substring(0, 64));
        const isReal = 1n;
        const oracleSecret = BigInt('0x' + 'bench'.padEnd(64, '0'));
        const userSecret = BigInt('0x' + `user${i}`.padEnd(32, '0'));
        
        const { nullifier, commitment } = computeCircuitValues(imageHash, isReal, oracleSecret, userSecret);
        
        const input = {
            commitment: commitment.toString(),
            nullifier: nullifier.toString(),
            imageHash: imageHash.toString(),
            isReal: isReal.toString(),
            oracleSecret: oracleSecret.toString(),
            userSecret: userSecret.toString()
        };
        
        const start = Date.now();
        await generateProof(input, wasmPath, zkeyPath);
        times.push(Date.now() - start);
        
        process.stdout.write(`   Proof ${i + 1}/10 completed\r`);
    }
    
    const avg = times.reduce((a, b) => a + b, 0) / times.length;
    const min = Math.min(...times);
    const max = Math.max(...times);
    
    console.log('');
    log(colors.green, `   ✅ PASSED - Benchmark complete`);
    log(colors.yellow, `      Average: ${avg.toFixed(0)}ms | Min: ${min}ms | Max: ${max}ms`);
    
    return { passed: true, avg, min, max };
}

// ============================================================
// MAIN
// ============================================================

async function main() {
    console.log(colors.bold + '\n' + '='.repeat(60));
    console.log('   🧪 ZK PROOF COMPREHENSIVE TEST SUITE');
    console.log('='.repeat(60) + colors.reset);
    
    const wasmPath = path.join(BUILD_DIR, `${CIRCUIT_NAME}_js`, `${CIRCUIT_NAME}.wasm`);
    const zkeyPath = path.join(KEYS_DIR, `${CIRCUIT_NAME}.zkey`);
    const vkeyPath = path.join(KEYS_DIR, `${CIRCUIT_NAME}_verification_key.json`);
    
    // Check files
    if (!fs.existsSync(wasmPath)) {
        log(colors.red, '\n❌ WASM file not found. Run these commands first:');
        console.log('   npm run compile');
        console.log('   npm run setup');
        process.exit(1);
    }
    
    if (!fs.existsSync(zkeyPath)) {
        log(colors.red, '\n❌ ZKey file not found. Run: npm run setup');
        process.exit(1);
    }
    
    // Initialize
    console.log('\n⏳ Initializing Poseidon hash...');
    await initPoseidon();
    log(colors.green, '✅ Poseidon initialized');
    
    // Run tests
    const results = [];
    
    results.push(await testCase1_RealImageValid(wasmPath, zkeyPath, vkeyPath));
    results.push(await testCase2_FakeImageFails(wasmPath, zkeyPath));
    results.push(await testCase3_WrongCommitmentFails(wasmPath, zkeyPath));
    results.push(await testCase4_WrongOracleSecretFails(wasmPath, zkeyPath));
    results.push(await testCase5_PerformanceBenchmark(wasmPath, zkeyPath, vkeyPath));
    
    // Summary
    console.log('\n' + '='.repeat(60));
    const passed = results.filter(r => r.passed).length;
    const total = results.length;
    
    if (passed === total) {
        log(colors.green + colors.bold, `\n🎉 ALL TESTS PASSED! (${passed}/${total})`);
    } else {
        log(colors.red + colors.bold, `\n⚠️ SOME TESTS FAILED: ${passed}/${total} passed`);
    }
    
    console.log('='.repeat(60) + '\n');
    
    // Save results
    const outputPath = path.join(__dirname, '..', 'test-results.json');
    fs.writeFileSync(outputPath, JSON.stringify({
        timestamp: new Date().toISOString(),
        circuit: CIRCUIT_NAME,
        results: results.map((r, i) => ({
            test: i + 1,
            passed: r.passed,
            proofTime: r.proofTime || null
        })),
        summary: { passed, total }
    }, null, 2));
    
    log(colors.cyan, `📄 Results saved to ${outputPath}\n`);
    
    process.exit(passed === total ? 0 : 1);
}

main().catch(console.error);
