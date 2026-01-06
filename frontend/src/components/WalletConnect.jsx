import './WalletConnect.css'

function WalletConnect({ account, userDID, onConnect, onDisconnect, onRegisterDID, loading, stats }) {

  const formatAddress = (address) => {
    if (!address) return '';
    return `${address.substring(0, 6)}...${address.substring(address.length - 4)}`;
  };

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
          <p className="connect-desc">Connect your MetaMask to establish your decentralized identity and secure verifictions.</p>
          <button className="btn btn-primary" onClick={onConnect}>
            Connect MetaMask
          </button>

          {stats && (
            <div className="stats-mini-grid" style={{ width: '100%', marginTop: '40px', opacity: 0.7 }}>
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
          <div className="user-card">
            <div className="user-header">
              <div className="user-avatar">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
                </svg>
              </div>
              <div className="user-info">
                <span className="user-label">Connected As</span>
                <span className="user-address">{formatAddress(account)}</span>
              </div>
            </div>
            <button className="disconnect-btn" onClick={onDisconnect}>
              Disconnect Wallet
            </button>
          </div>

          <div className={`did-status-card ${userDID ? 'active' : ''}`}>
            <div className="did-header">
              <span className="did-title">
                <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                Digital ID
              </span>
              <span className={`status-badge ${userDID ? 'verified' : 'unverified'}`}>
                {userDID ? 'Verified' : 'Unregistered'}
              </span>
            </div>

            {userDID ? (
              <>
                <div className="did-value">{userDID.did}</div>
                <div style={{ fontSize: '0.8rem', color: 'var(--success)', marginTop: '8px' }}>Unique Identity Active</div>
              </>
            ) : (
              <>
                <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: '16px' }}>
                  Register a DID to enable blockchain recording of your AI verifications.
                </p>
                <button className="btn btn-primary" style={{ width: '100%' }} onClick={onRegisterDID} disabled={loading}>
                  {loading ? 'Registering...' : 'Register Identity'}
                </button>
              </>
            )}
          </div>

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
  );
}

export default WalletConnect;
