/**
 * Script: Trusted Setup (Powers of Tau + Phase 2)
 * 
 * Tạo proving key và verification key
 * CHÚ Ý: Trong production, cần ceremony với nhiều participants!
 */

const snarkjs = require('snarkjs');
const fs = require('fs');
const path = require('path');

const CIRCUIT_NAME = 'simple_proof';
const BUILD_DIR = path.join(__dirname, '..', 'build');
const KEYS_DIR = path.join(__dirname, '..', 'keys');

async function setup() {
    console.log('🔐 Starting Trusted Setup...\n');
    
    // Tạo keys directory
    if (!fs.existsSync(KEYS_DIR)) {
        fs.mkdirSync(KEYS_DIR, { recursive: true });
    }
    
    const r1csPath = path.join(BUILD_DIR, `${CIRCUIT_NAME}.r1cs`);
    
    if (!fs.existsSync(r1csPath)) {
        console.error('❌ R1CS file not found. Run compile first!');
        console.log('   npm run compile');
        process.exit(1);
    }
    
    try {
        // ===== PHASE 1: Powers of Tau =====
        console.log('📜 Phase 1: Powers of Tau ceremony...');
        
        const ptauPath = path.join(KEYS_DIR, 'pot12_final.ptau');
        
        // Kiểm tra nếu đã có ptau file (có thể download sẵn)
        if (!fs.existsSync(ptauPath)) {
            console.log('   Generating new Powers of Tau (this may take a while)...');
            
            // Bắt đầu ceremony
            await snarkjs.powersOfTau.newAccumulator(
                snarkjs.curves.bn128, 
                12,  // 2^12 = 4096 constraints max
                path.join(KEYS_DIR, 'pot12_0000.ptau')
            );
            
            // Contribute (trong production cần nhiều người contribute)
            await snarkjs.powersOfTau.contribute(
                path.join(KEYS_DIR, 'pot12_0000.ptau'),
                path.join(KEYS_DIR, 'pot12_0001.ptau'),
                'First contribution',
                'random-entropy-' + Date.now()
            );
            
            // Prepare phase 2
            await snarkjs.powersOfTau.preparePhase2(
                path.join(KEYS_DIR, 'pot12_0001.ptau'),
                ptauPath
            );
            
            // Cleanup intermediate files
            fs.unlinkSync(path.join(KEYS_DIR, 'pot12_0000.ptau'));
            fs.unlinkSync(path.join(KEYS_DIR, 'pot12_0001.ptau'));
            
            console.log('   ✅ Powers of Tau completed\n');
        } else {
            console.log('   ✅ Using existing Powers of Tau file\n');
        }
        
        // ===== PHASE 2: Circuit-specific setup =====
        console.log('📜 Phase 2: Circuit-specific setup...');
        
        const zkeyPath = path.join(KEYS_DIR, `${CIRCUIT_NAME}.zkey`);
        const vkeyPath = path.join(KEYS_DIR, `${CIRCUIT_NAME}_verification_key.json`);
        
        // Generate initial zkey
        await snarkjs.zKey.newZKey(
            r1csPath,
            ptauPath,
            path.join(KEYS_DIR, `${CIRCUIT_NAME}_0000.zkey`)
        );
        
        // Contribute to phase 2
        await snarkjs.zKey.contribute(
            path.join(KEYS_DIR, `${CIRCUIT_NAME}_0000.zkey`),
            zkeyPath,
            'Deepfake Verification Contribution',
            'entropy-' + Math.random().toString(36)
        );
        
        // Cleanup
        fs.unlinkSync(path.join(KEYS_DIR, `${CIRCUIT_NAME}_0000.zkey`));
        
        console.log('   ✅ Proving key generated\n');
        
        // Export verification key
        console.log('📤 Exporting verification key...');
        const vkey = await snarkjs.zKey.exportVerificationKey(zkeyPath);
        fs.writeFileSync(vkeyPath, JSON.stringify(vkey, null, 2));
        
        console.log('\n✅ Trusted Setup completed!');
        console.log('\nGenerated files:');
        console.log(`  - ${zkeyPath} (proving key - KEEP SECRET)`);
        console.log(`  - ${vkeyPath} (verification key - PUBLIC)`);
        
    } catch (error) {
        console.error('❌ Setup failed:', error);
        process.exit(1);
    }
}

setup();
