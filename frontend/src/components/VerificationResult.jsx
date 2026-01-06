import './VerificationResult.css'

// Tự động chọn block explorer dựa trên Chain ID
const getExplorerUrl = (txHash) => {
  const chainId = import.meta.env.VITE_CHAIN_ID;
  const explorers = {
    '11155111': `https://sepolia.etherscan.io/tx/${txHash}`,
    '80002': `https://amoy.polygonscan.com/tx/${txHash}`,
    '31337': null // Hardhat local không có explorer
  };
  return explorers[chainId] || null;
};

function VerificationResult({ result }) {
  const isReal = result.label === 'REAL';
  const confidence = (result.confidence * 100).toFixed(1);

  return (
    <div className={`verification-result ${isReal ? 'real' : 'fake'}`}>
      <div className="result-header">
        <div className="result-icon">
          {isReal ? (
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
            </svg>
          )}
        </div>
        <div className="result-title">
          <h3 className="brand-font">{isReal ? 'Authentic Media Verified' : 'Deepfake Manipulation Detected'}</h3>
          <p>{isReal ? 'No artificial manipulation traces found in image spectrum analysis.' : 'High probability of AI-generated artifacts or facial manipulation detected.'}</p>
        </div>
      </div>

      <div className="result-body">
        {/* Confidence Data */}
        <div className="confidence-wrapper">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span className="meta-label">AI Confidence Score</span>
            <span className="meta-value" style={{ fontSize: '1rem' }}>{confidence}%</span>
          </div>
          <div className="conf-track">
            <div className="conf-fill" style={{ width: `${confidence}%` }}></div>
          </div>
        </div>

        <div className="meta-grid">
          <div className="meta-item">
            <span className="meta-label">Real Probability</span>
            <span className="meta-value" style={{ color: isReal ? 'var(--success)' : 'var(--text-muted)' }}>
              {(result.real_prob * 100).toFixed(2)}%
            </span>
          </div>
          <div className="meta-item">
            <span className="meta-label">Fake Probability</span>
            <span className="meta-value" style={{ color: !isReal ? 'var(--danger)' : 'var(--text-muted)' }}>
              {(result.fake_prob * 100).toFixed(2)}%
            </span>
          </div>
        </div>

        {/* Blockchain Terminal */}
        <div className="blockchain-proof">
          <div className="proof-header">
            <span className="proof-title">
              <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
              Immutable Ledger Record
            </span>
            <span className={`proof-status ${result.onChain ? 'success' : 'pending'}`}>
              {result.onChain ? 'CONFIRMED' : 'PENDING'}
            </span>
          </div>

          <div className="proof-content">
            <div className="tx-row">
              <span className="tx-label">HASH:</span>
              <span className="tx-val">{result.image_hash ? result.image_hash.substring(0, 32) + '...' : 'Calculating...'}</span>
            </div>

            {result.onChain ? (
              <div className="tx-row">
                <span className="tx-label">TX ID:</span>
                <span className="tx-val" style={{ color: 'var(--primary)' }}>
                  {result.transactionHash.substring(0, 20)}...
                  {getExplorerUrl(result.transactionHash) && (
                    <a href={getExplorerUrl(result.transactionHash)} target="_blank" rel="noopener noreferrer" className="tx-link">VIEW</a>
                  )}
                </span>
              </div>
            ) : (
              <div className="tx-row">
                <span className="tx-label">STATUS:</span>
                <span className="tx-val" style={{ color: 'var(--text-dim)' }}>Waiting for DID signature...</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default VerificationResult;
