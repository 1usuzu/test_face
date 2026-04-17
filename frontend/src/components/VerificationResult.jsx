import './VerificationResult.css'

const CONSUMER_APP_URL = import.meta.env.VITE_CONSUMER_APP_URL || 'http://localhost:8001'

const getExplorerUrl = (txHash) => {
  const chainId = import.meta.env.VITE_CHAIN_ID
  const map = {
    '11155111': `https://sepolia.etherscan.io/tx/${txHash}`,
    '80002': `https://amoy.polygonscan.com/tx/${txHash}`,
    '31337': null,
  }
  return map[chainId] ?? null
}

const RISK_COLOR = {
  low: 'var(--success)',
  medium: 'var(--warn)',
  high: 'var(--danger)',
  critical: '#7c3aed',
}

/** Display bands (fake_probability). Backend decision threshold remains 0.65. */
const BAND_REVIEW_MIN = 0.4
const BAND_FAKE_MIN = 0.65

/**
 * @returns {'likely_authentic' | 'review_needed' | 'fake_detected'}
 */
function tierFromFakeProb(fakeProb) {
  const p = Number(fakeProb)
  if (!Number.isFinite(p)) return 'review_needed'
  if (p < BAND_REVIEW_MIN) return 'likely_authentic'
  if (p >= BAND_FAKE_MIN) return 'fake_detected'
  return 'review_needed'
}

const TIER_UI = {
  likely_authentic: {
    title: 'Likely Authentic',
    subtitle:
      'Fake probability is below 40%. The image is consistent with authentic media under this policy.',
    barLabel: 'Real likelihood',
    barFraction: (realProb) => realProb,
  },
  review_needed: {
    title: 'Review Needed / Suspicious',
    subtitle:
      'Fake probability is between 40% and 65%. Treat as suspicious; additional review is recommended.',
    barLabel: 'Fake likelihood (suspicious band)',
    barFraction: (_realProb, fakeProb) => fakeProb,
  },
  fake_detected: {
    title: 'Fake Detected',
    subtitle:
      'Fake probability is at or above 65% — strong indication of manipulation or synthetic content.',
    barLabel: 'Fake likelihood',
    barFraction: (_realProb, fakeProb) => fakeProb,
  },
}

function CopyButton({ text }) {
  const handleCopy = () => {
    navigator.clipboard.writeText(text).then(() => alert('Link copied to clipboard!'))
  }
  return (
    <button
      onClick={handleCopy}
      title="Copy verification link"
      style={{
        padding: '4px 10px', fontSize: '.75rem', fontWeight: 600,
        color: 'var(--primary)', background: 'rgba(37,99,235,.1)',
        border: 'none', borderRadius: '6px', cursor: 'pointer', flexShrink: 0,
      }}
    >
      Copy
    </button>
  )
}

function VerificationResult({ result }) {
  const statusVal = result.status
  const isNonClassifiable =
    result.label === 'ERROR' || (statusVal != null && statusVal !== 'ok')

  const realProb = Number(result.real_prob ?? 0)
  const fakeProb = Number(result.fake_prob ?? 0)

  const tier = isNonClassifiable ? null : tierFromFakeProb(fakeProb)
  const tierUi = tier ? TIER_UI[tier] : null

  const cardTone = isNonClassifiable
    ? 'unavailable'
    : tier === 'likely_authentic'
      ? 'real'
      : tier === 'review_needed'
        ? 'suspicious'
        : 'fake'

  const barFraction = tierUi ? tierUi.barFraction(realProb, fakeProb) : 0
  const barPct = (Math.max(0, Math.min(1, barFraction)) * 100).toFixed(1)

  const riskLevel =
    result.risk_level ||
    (isNonClassifiable ? 'unknown' : tier === 'likely_authentic' ? 'low' : tier === 'review_needed' ? 'medium' : 'high')
  const riskColor = RISK_COLOR[riskLevel] || 'var(--text-muted)'

  // verification_link: either from API result, or construct from consumer app URL + image_hash
  const verificationLink = result.verification_link
    || (result.image_hash ? `${CONSUMER_APP_URL}/verify/${result.image_hash}` : null)

  const downloadZkpFiles = () => {
    if (!result.zkp || result.zkp.error) return;

    const proofData = JSON.stringify(result.zkp.proof, null, 2);
    const publicData = JSON.stringify(result.zkp.publicSignals, null, 2);

    const downloadFile = (filename, content) => {
      const blob = new Blob([content], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    };

    downloadFile('proof.json', proofData);
    setTimeout(() => downloadFile('public.json', publicData), 100);
  };

  return (
    <div className={`verification-result ${cardTone}`}>
      <div className="result-header">
        <div className="result-icon">
          {isNonClassifiable ? (
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
            </svg>
          ) : tier === 'likely_authentic' ? (
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          ) : tier === 'review_needed' ? (
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
            </svg>
          )}
        </div>
        <div className="result-title">
          <h3 className="brand-font">
            {isNonClassifiable
              ? 'Verification Unavailable'
              : tierUi?.title}
          </h3>
          <p>
            {isNonClassifiable
              ? (result.message || 'The image could not be classified safely. Try a clearer face photo.')
              : tierUi?.subtitle}
          </p>
        </div>
      </div>

      <div className="result-body">
        {/* Confidence */}
        <div className="confidence-wrapper">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span className="meta-label">{tierUi ? tierUi.barLabel : 'Score'}</span>
            <span className="meta-value" style={{ fontSize: '1rem' }}>{isNonClassifiable ? '—' : `${barPct}%`}</span>
          </div>
          <div className="conf-track">
            <div className="conf-fill" style={{ width: `${isNonClassifiable ? 0 : barPct}%` }}></div>
          </div>
          {!isNonClassifiable && (
            <p className="tier-band-hint">
              Bands: &lt;40% likely authentic · 40–65% review · ≥65% fake (same as model decision threshold)
            </p>
          )}
        </div>

        {/* Probability grid */}
        <div className="meta-grid">
          <div className="meta-item">
            <span className="meta-label">Real Probability</span>
            <span
              className="meta-value"
              style={{
                color:
                  tier === 'likely_authentic' ? 'var(--success)' : 'var(--text-muted)',
              }}
            >
              {(realProb * 100).toFixed(2)}%
            </span>
          </div>
          <div className="meta-item">
            <span className="meta-label">Fake Probability</span>
            <span
              className="meta-value"
              style={{
                color:
                  tier === 'fake_detected'
                    ? 'var(--danger)'
                    : tier === 'review_needed'
                      ? 'var(--warn)'
                      : 'var(--text-muted)',
              }}
            >
              {(fakeProb * 100).toFixed(2)}%
            </span>
          </div>
          <div className="meta-item">
            <span className="meta-label">Risk Level</span>
            <span className="meta-value" style={{ color: riskColor, textTransform: 'uppercase', fontSize: '.85rem' }}>
              {riskLevel}
            </span>
          </div>
          <div className="meta-item">
            <span className="meta-label">Detector</span>
            <span className="meta-value" style={{ fontSize: '.8rem', color: 'var(--text-muted)' }}>
              {result.detector_version || 'ensemble'}
            </span>
          </div>
        </div>

        {/* Portal Link — Xem trên Consumer Portal */}
        {verificationLink && (
          <div style={{
            display: 'flex', flexDirection: 'column', gap: '8px',
            background: 'rgba(37,99,235,.05)', border: '1px solid rgba(37,99,235,.15)',
            borderRadius: '9px', padding: '12px 14px', marginBottom: '12px',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
              <svg width="16" height="16" fill="none" stroke="var(--primary)" viewBox="0 0 24 24" style={{ flexShrink: 0 }}>
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
              </svg>
              <span style={{ fontSize: '.85rem', fontWeight: 600, color: 'var(--primary)' }}>
                Xem trên Portal
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
              <a
                href={verificationLink}
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  fontSize: '.78rem', color: 'var(--primary)', wordBreak: 'break-all', flex: 1,
                  textDecoration: 'none', opacity: 0.9,
                }}
              >
                {verificationLink}
              </a>
              <CopyButton text={verificationLink} />
            </div>
            {result.transactionHash && (
              <div style={{ fontSize: '.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                Hoặc tra cứu theo TX: <a
                  href={`${CONSUMER_APP_URL}?tx=${result.transactionHash}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ color: 'var(--primary)', textDecoration: 'none' }}
                >
                  {result.transactionHash.substring(0, 20)}…
                </a>
              </div>
            )}
          </div>
        )}

        {/* Blockchain record */}
        <div className="blockchain-proof">
          <div className="proof-header">
            <span className="proof-title">
              <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
              Immutable Ledger Record
            </span>
            <span className={`proof-status ${result.onChain ? 'success' : 'pending'}`}>
              {result.onChain ? 'CONFIRMED' : (result.onChainError ? 'FAILED' : 'PENDING')}
            </span>
          </div>

          <div className="proof-content">
            <div className="tx-row">
              <span className="tx-label">HASH:</span>
              <span className="tx-val">
                {result.image_hash ? `${result.image_hash.substring(0, 32)}…` : 'Calculating...'}
              </span>
            </div>

            {result.onChain ? (
              <div className="tx-row">
                <span className="tx-label">TX ID:</span>
                <span className="tx-val" style={{ color: 'var(--primary)' }}>
                  {result.transactionHash.substring(0, 20)}…
                  {getExplorerUrl(result.transactionHash) && (
                    <a href={getExplorerUrl(result.transactionHash)} target="_blank" rel="noopener noreferrer" className="tx-link">VIEW</a>
                  )}
                </span>
              </div>
            ) : (
              <div className="tx-row">
                <span className="tx-label">STATUS:</span>
                <span className="tx-val" style={{ color: result.onChainError ? 'var(--danger)' : 'var(--text-dim)' }}>
                  {result.onChainError || result.pendingReason || 'Not yet submitted to blockchain'}
                </span>
              </div>
            )}
          </div>
        </div>

        {result.zkp && (
          <div style={{
            display: 'flex', flexDirection: 'column', gap: '8px',
            marginTop: '10px'
          }}>
            <div style={{
              padding: '10px 13px',
              background: 'rgba(124,58,237,.05)', border: '1px solid rgba(124,58,237,.15)',
              borderRadius: '9px', fontSize: '.82rem', color: '#7c3aed',
              display: 'flex', alignItems: 'center', gap: '7px',
            }}>
              <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24" style={{ flexShrink: 0 }}>
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" />
              </svg>
              {result.zkp.error
                ? `ZK Proof failed: ${result.zkp.error}`
                : `ZK Proof generated — commitment: ${String(result.zkp.commitment).substring(0, 20)}…`
              }
            </div>

            {!result.zkp.error && (
              <button
                onClick={downloadZkpFiles}
                style={{
                  background: '#7c3aed', color: '#fff', border: 'none',
                  padding: '8px 12px', borderRadius: '6px', cursor: 'pointer',
                  fontWeight: 600, fontSize: '0.85rem', alignSelf: 'flex-start',
                  display: 'flex', alignItems: 'center', gap: '6px'
                }}
              >
                <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Tải Bằng Chứng ZKP (proof.json & public.json)
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default VerificationResult

