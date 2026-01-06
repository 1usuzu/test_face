/**
 * Script: Export Solidity Verifier Contract
 */

const snarkjs = require('snarkjs');
const fs = require('fs');
const path = require('path');

const CIRCUIT_NAME = 'simple_proof';
const KEYS_DIR = path.join(__dirname, '..', 'keys');
const CONTRACTS_DIR = path.join(__dirname, '..', 'contracts');

async function exportVerifier() {
    console.log('📤 Exporting Solidity Verifier Contract...\n');
    
    const zkeyPath = path.join(KEYS_DIR, `${CIRCUIT_NAME}.zkey`);
    
    if (!fs.existsSync(zkeyPath)) {
        console.error('❌ Zkey not found. Run setup first!');
        process.exit(1);
    }
    
    // Create contracts directory
    if (!fs.existsSync(CONTRACTS_DIR)) {
        fs.mkdirSync(CONTRACTS_DIR, { recursive: true });
    }
    
    try {
        // Export verifier
        const verifierCode = await snarkjs.zKey.exportSolidityVerifier(
            zkeyPath,
            {
                groth16: fs.readFileSync(
                    path.join(__dirname, 'templates', 'groth16_verifier.sol.ejs'),
                    'utf8'
                ).toString()
            }
        );
        
        const outputPath = path.join(CONTRACTS_DIR, 'Groth16Verifier.sol');
        fs.writeFileSync(outputPath, verifierCode);
        
        console.log(`✅ Verifier contract exported to ${outputPath}`);
        console.log('\nNext steps:');
        console.log('1. Copy Groth16Verifier.sol to blockchain/contracts/');
        console.log('2. Import and use in DeepfakeVerification.sol');
        
    } catch (error) {
        // Nếu không có template, dùng cách khác
        console.log('Using built-in template...');
        
        const { execSync } = require('child_process');
        const cmd = `npx snarkjs zkey export solidityverifier "${zkeyPath}" "${CONTRACTS_DIR}/Groth16Verifier.sol"`;
        
        try {
            execSync(cmd, { stdio: 'inherit' });
            console.log(`\n✅ Verifier contract exported!`);
        } catch (e) {
            console.error('❌ Export failed. Try manually:');
            console.log(`   ${cmd}`);
        }
    }
}

exportVerifier();
