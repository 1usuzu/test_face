import './WalletConnect.css'

function WalletConnect({ account, userDID, onConnect, onDisconnect, onRegisterDID, loading, stats }) {

  const formatAddress = (address) => {
    if (!address) return ''
    return `${address.substring(0, 6)}…${address.substring(address.length - 4)}`
  }

  const copyDID = () => {
    if (userDID?.did) {
      navigator.clipboard.writeText(userDID.did).then(() => alert('DID copied to clipboard!'))
    }
  }

  return (
    <div className="wallet-connect">
      {!account ? (
        <div className="connect-section glass-panel">
          <div className="connect-icon-wrapper">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M21 12a2.25 2.25 0 00-2.25-2.25H15a3 3 0 11-6 0H5.25A2.25 2.25 0 003 12m18 0v6a2.25 2.25 0 01-2.25 2.25H5.25A2.25 2.25 0 013 18v-6m18 0V9M3 12V9m18 0a2.25 2.25 0 00-2.25-2.25H5.25A2.25 2.25 0 013 9m18 0V6a2.25 2.25 0 00-2.25-2.25H5.25A2.25 2.25 0 013 6v3" />
            </svg>
          </div>
          <h3 className="connect-title brand-font">Connect Wallet</h3>
          <p className="connect-desc">Connect your MetaMask to establish your decentralized identity and enable blockchain-verified results.</p>
          <button className="btn btn-primary" onClick={onConnect}>
            Connect MetaMask
          </button>

          {stats && (
            <div className="stats-mini-grid" style={{ width: '100%', marginTop: '32px' }}>
              <div className="stat-item">
                <span className="stat-val">{stats.totalDIDs || 0}</span>
                <span className="stat-lbl">Identities</span>
              </div>
              <div className="stat-item">
                <span className="stat-val">{stats.totalVerifications || 0}</span>
                <span className="stat-lbl">Verifications</span>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="connected-section glass-panel">
          {/* ── Wallet card ── */}
          <div className="user-card">
            <div className="user-header">
              <div className="user-avatar">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
                </svg>
              </div>
              <div className="user-info">
                <span className="user-label">Connected Wallet</span>
                <span className="user-address">{formatAddress(account)}</span>
              </div>
            </div>
            <button className="disconnect-btn" onClick={onDisconnect}>Disconnect</button>
          </div>

          {/* ── DID card ── */}
          <div className={`did-status-card ${userDID ? 'active' : ''}`}>
            <div className="did-header">
              <span className="did-title">
                <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Decentralized Identity
              </span>
              <span className={`status-badge ${userDID ? 'verified' : 'unverified'}`}>
                {userDID ? '✓ Active' : 'Not registered'}
              </span>
            </div>

            {userDID ? (
              <>
                <div className="did-value">{userDID.did}</div>
                <div style={{ display: 'flex', gap: '8px', marginTop: '10px' }}>
                  <button
                    onClick={copyDID}
                    style={{
                      flex: 1, padding: '8px 0', fontSize: '0.82rem', fontWeight: 600,
                      color: 'var(--primary)', background: 'rgba(37,99,235,.1)',
                      border: 'none', borderRadius: '7px', cursor: 'pointer',
                    }}
                  >
                    Copy DID
                  </button>
                </div>
                <div style={{ fontSize: '.75rem', color: 'var(--success)', marginTop: '8px', textAlign: 'center' }}>
                  ● Identity active — on-chain verification enabled
                </div>
              </>
            ) : (
              <>
                <p style={{ fontSize: '.85rem', color: 'var(--text-muted)', marginBottom: '14px' }}>
                  Register a DID to enable blockchain recording of your AI verifications and share verification links.
                </p>

                {/* Benefit list */}
                <ul style={{ fontSize: '.8rem', color: 'var(--text-muted)', listStyle: 'none', marginBottom: '16px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  {[
                    'Record results permanently on-chain',
                    'Generate public verification links',
                    'Issue Verifiable Credentials',
                  ].map(b => (
                    <li key={b} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <span style={{ color: 'var(--success)', fontWeight: 700 }}>✓</span> {b}
                    </li>
                  ))}
                </ul>

                <button
                  className="btn btn-primary"
                  style={{ width: '100%' }}
                  onClick={onRegisterDID}
                  disabled={loading}
                >
                  {loading ? 'Registering…' : 'Register Identity (Free)'}
                </button>
              </>
            )}
          </div>

          {/* ── Platform stats ── */}
          {stats && (
            <div className="stats-mini-grid">
              <div className="stat-item">
                <span className="stat-val">{stats.totalDIDs}</span>
                <span className="stat-lbl">Identities</span>
              </div>
              <div className="stat-item">
                <span className="stat-val">{stats.totalVerifications}</span>
                <span className="stat-lbl">Verifications</span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default WalletConnect
