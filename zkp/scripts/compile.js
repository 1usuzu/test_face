/**
 * Script: Compile Circom circuits
 * 
 * Biên dịch circuit thành R1CS, WASM, và các file cần thiết
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const CIRCUIT_NAME = 'simple_proof';  // Dùng circuit đơn giản trước
const CIRCUITS_DIR = path.join(__dirname, '..', 'circuits');
const BUILD_DIR = path.join(__dirname, '..', 'build');

async function compile() {
    console.log('🔧 Compiling Circom circuit...\n');
    
    // Tạo build directory
    if (!fs.existsSync(BUILD_DIR)) {
        fs.mkdirSync(BUILD_DIR, { recursive: true });
    }
    
    const circuitPath = path.join(CIRCUITS_DIR, `${CIRCUIT_NAME}.circom`);
    
    if (!fs.existsSync(circuitPath)) {
        console.error(`❌ Circuit not found: ${circuitPath}`);
        process.exit(1);
    }
    
    try {
        // Compile với circom
        console.log(`📄 Compiling ${CIRCUIT_NAME}.circom...`);
        
        const cmd = `circom "${circuitPath}" --r1cs --wasm --sym --c -o "${BUILD_DIR}"`;
        console.log(`> ${cmd}\n`);
        
        execSync(cmd, { stdio: 'inherit' });
        
        console.log('\n✅ Compilation successful!');
        console.log('\nGenerated files:');
        console.log(`  - ${BUILD_DIR}/${CIRCUIT_NAME}.r1cs (constraints)`);
        console.log(`  - ${BUILD_DIR}/${CIRCUIT_NAME}_js/ (WASM for JS)`);
        console.log(`  - ${BUILD_DIR}/${CIRCUIT_NAME}.sym (debug symbols)`);
        
    } catch (error) {
        console.error('❌ Compilation failed:', error.message);
        console.log('\n💡 Make sure circom is installed:');
        console.log('   npm install -g circom');
        console.log('   Or: cargo install --path circom (from source)');
        process.exit(1);
    }
}

compile();
