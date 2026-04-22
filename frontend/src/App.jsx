import { useEffect, useState } from 'react'
import { ethers } from 'ethers'
import WalletConnect from './components/WalletConnect'
import ImageUpload from './components/ImageUpload'
import VerificationResult from './components/VerificationResult'
import BrowserZKProof from './lib/browser-proof'
import './App.css'

import { Shield, ScanFace, KeyRound } from 'lucide-react'

const CONTRACT_ABI = [
  "function registerDID(string calldata _did, string calldata _publicKeyBase58) external",
  "function recordVerification(bytes32 _imageHash, bool _isReal, uint256 _confidence, bytes calldata _signature) external",
  "function getVerification(bytes32 _imageHash) external view returns (tuple(bytes32 imageHash, string subjectDid, string issuerDid, bool isReal, uint256 confidence, uint256 timestamp, bytes32 credentialHash))",
  "function didDocuments(address) external view returns (address owner, string did, string publicKeyBase58, bool isActive, uint256 createdAt, uint256 updatedAt)",
  "function getStats() external view returns (uint256, uint256)"
];

const CONTRACT_ADDRESS = import.meta.env.VITE_CONTRACT_ADDRESS || "0x_YOUR_CONTRACT_ADDRESS";
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const EXPECTED_CHAIN_ID = Number(import.meta.env.VITE_CHAIN_ID || 31337);
const EXPECTED_NETWORK_NAME = import.meta.env.VITE_NETWORK_NAME || 'Hardhat Local';
const EXPECTED_RPC_URL = import.meta.env.VITE_RPC_URL || 'http://127.0.0.1:8545';

const parseChainId = (chainIdHexOrNumber) => {
  if (typeof chainIdHexOrNumber === 'number') return chainIdHexOrNumber;
  if (typeof chainIdHexOrNumber === 'string') {
    return chainIdHexOrNumber.startsWith('0x')
      ? Number.parseInt(chainIdHexOrNumber, 16)
      : Number.parseInt(chainIdHexOrNumber, 10);
  }
  return NaN;
};

const normalizeDidDocument = (didDoc) => {
  if (!didDoc) return null;
  return {
    owner: didDoc.owner,
    did: didDoc.did,
    publicKeyBase58: didDoc.publicKeyBase58,
    isActive: Boolean(didDoc.isActive),
    createdAt: Number(didDoc.createdAt ?? 0n),
    updatedAt: Number(didDoc.updatedAt ?? 0n)
  };
};

const isUserRejectedError = (error) => {
  const code = error?.code ?? error?.info?.error?.code;
  const reason = error?.reason;
  return code === 4001 || code === 'ACTION_REJECTED' || reason === 'rejected';
};

const formatTxError = (actionLabel, error) => {
  if (isUserRejectedError(error)) {
    return `Bạn đã hủy giao dịch ${actionLabel} trên MetaMask. Không có thay đổi nào được ghi lên blockchain.`;
  }
  const detail = error?.shortMessage || error?.reason || error?.info?.error?.message || error?.message || 'Không xác định';
  return `${actionLabel} thất bại: ${detail}`;
};

const formatWalletActionError = (actionLabel, error) => {
  if (isUserRejectedError(error)) {
    return `Bạn đã hủy thao tác ${actionLabel} trên MetaMask.`;
  }
  const detail = error?.shortMessage || error?.reason || error?.info?.error?.message || error?.message || 'Không xác định';
  return `${actionLabel} thất bại: ${detail}`;
};

/**
 * `/api/verify-zkp` now returns `fake_prob` and `real_prob` (parity with `/api/verify`).
 * Legacy fallback: if an older server omits probabilities, derive them from `label` +
 * post-PR2 `confidence` (class certainty), never by assuming `fake_prob === confidence`.
 */
function normalizeVerifyZkpResponse(result) {
  if (typeof result.fake_prob === 'number' && typeof result.real_prob === 'number') {
    return { ...result };
  }
  const c = Number(result.confidence ?? 0);
  if (result.label === 'REAL') {
    return { ...result, real_prob: c, fake_prob: 1 - c };
  }
  if (result.label === 'FAKE') {
    return { ...result, fake_prob: c, real_prob: 1 - c };
  }
  return { ...result };
}

function App() {
  const [account, setAccount] = useState(null);
  const [contract, setContract] = useState(null);
  const [userDID, setUserDID] = useState(null);
  const [verificationResult, setVerificationResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [stats, setStats] = useState({ totalDIDs: 0, totalVerifications: 0 });

  const submitVerificationOnChain = async (apiResult, currentResult) => {
    if (!contract) {
      return { ...currentResult, onChain: false, pendingReason: 'Chưa kết nối contract' };
    }
    if (!userDID?.isActive) {
      return { ...currentResult, onChain: false, pendingReason: 'Cần đăng ký DID trước khi ghi blockchain' };
    }
    if (!apiResult?.signature) {
      return { ...currentResult, onChain: false, pendingReason: 'Không có chữ ký oracle để ghi on-chain' };
    }

    try {
      setLoadingMessage('Đang ghi kết quả xác thực lên blockchain...');

      const imageHashBytes32 = apiResult.image_hash?.startsWith('0x')
        ? apiResult.image_hash
        : `0x${apiResult.image_hash}`;
      const oracleSignature = apiResult.signature?.startsWith('0x')
        ? apiResult.signature
        : `0x${apiResult.signature || ''}`;
      const isReal = apiResult.label === 'REAL';
      const confidenceOnChain = Math.round(Number(apiResult.confidence || 0) * 10000);

      if (!oracleSignature || oracleSignature === '0x') {
        return {
          ...currentResult,
          onChain: false,
          onChainError: 'Oracle signature không hợp lệ'
        };
      }

      const tx = await contract.recordVerification(
        imageHashBytes32,
        isReal,
        confidenceOnChain,
        oracleSignature
      );
      await tx.wait();

      const [dids, verifications] = await contract.getStats();
      setStats({ totalDIDs: Number(dids), totalVerifications: Number(verifications) });

      return {
        ...currentResult,
        onChain: true,
        transactionHash: tx.hash,
        pendingReason: null,
        onChainError: null
      };
    } catch (error) {
      return {
        ...currentResult,
        onChain: false,
        onChainError: formatTxError('ghi kết quả xác thực', error)
      };
    }
  };

  const ensureExpectedNetwork = async () => {
    const provider = new ethers.BrowserProvider(window.ethereum);
    const currentNetwork = await provider.getNetwork();
    const currentChainId = Number(currentNetwork.chainId);
    if (currentChainId === EXPECTED_CHAIN_ID) return provider;

    const targetChainHex = `0x${EXPECTED_CHAIN_ID.toString(16)}`;
    try {
      await window.ethereum.request({
        method: 'wallet_switchEthereumChain',
        params: [{ chainId: targetChainHex }]
      });
    } catch (switchError) {
      if (switchError?.code === 4902) {
        await window.ethereum.request({
          method: 'wallet_addEthereumChain',
          params: [{
            chainId: targetChainHex,
            chainName: EXPECTED_NETWORK_NAME,
            rpcUrls: [EXPECTED_RPC_URL],
            nativeCurrency: { name: 'Ether', symbol: 'ETH', decimals: 18 }
          }]
        });
        await window.ethereum.request({
          method: 'wallet_switchEthereumChain',
          params: [{ chainId: targetChainHex }]
        });
      } else {
        throw switchError;
      }
    }

    return new ethers.BrowserProvider(window.ethereum);
  };

  const connectWallet = async () => {
    if (!window.ethereum) {
      alert("Vui lòng cài đặt MetaMask!");
      return;
    }
    try {
      const provider = await ensureExpectedNetwork();
      const network = await provider.getNetwork();
      const currentChainId = Number(network.chainId);
      if (currentChainId !== EXPECTED_CHAIN_ID) {
        alert(`Sai network: đang ở chainId ${currentChainId}. Vui lòng chuyển sang ${EXPECTED_NETWORK_NAME} (chainId ${EXPECTED_CHAIN_ID}).`);
        return;
      }

      const accounts = await provider.send("eth_requestAccounts", []);
      const signer = await provider.getSigner();
      const contractCode = await provider.getCode(CONTRACT_ADDRESS);
      if (!contractCode || contractCode === '0x') {
        alert(`Không tìm thấy contract tại ${CONTRACT_ADDRESS} trên ${EXPECTED_NETWORK_NAME}. Hãy deploy smart contract đúng mạng rồi cập nhật lại VITE_CONTRACT_ADDRESS.`);
        return;
      }
      const contractInstance = new ethers.Contract(CONTRACT_ADDRESS, CONTRACT_ABI, signer);

      setAccount(accounts[0]);
      setContract(contractInstance);

      try {
        const didDoc = await contractInstance.didDocuments(accounts[0]);
        const normalizedDidDoc = normalizeDidDocument(didDoc);
        if (normalizedDidDoc?.isActive) setUserDID(normalizedDidDoc);
        const [dids, verifications] = await contractInstance.getStats();
        setStats({ totalDIDs: Number(dids), totalVerifications: Number(verifications) });
      } catch (e) { console.log("Init data load error (Contract might not be deployed yet)", e); }

    } catch (error) {
      console.error("Failed to connect wallet:", error);
      alert(formatWalletActionError('kết nối ví', error));
    }
  };

  const registerDID = async () => {
    if (!account) return alert("Vui lòng kết nối ví trước khi đăng ký DID!");
    if (!contract) return alert("Hệ thống bị mất kết nối với Smart Contract (thường do mạng Hardhat chưa đồng bộ). Vui lòng F5 làm mới lại trang và Kết Nối Ví lại lần nữa!");
    try {
      setLoading(true);

      const provider = new ethers.BrowserProvider(window.ethereum);
      const network = await provider.getNetwork();
      const currentChainId = Number(network.chainId);
      if (currentChainId !== EXPECTED_CHAIN_ID) {
        throw new Error(`Sai network. Hãy chuyển sang ${EXPECTED_NETWORK_NAME} (chainId ${EXPECTED_CHAIN_ID}).`);
      }

      const contractCode = await provider.getCode(CONTRACT_ADDRESS);
      if (!contractCode || contractCode === '0x') {
        throw new Error(`Contract chưa được deploy tại ${CONTRACT_ADDRESS} trên mạng hiện tại.`);
      }

      const randomId = Array.from(crypto.getRandomValues(new Uint8Array(16)))
        .map(b => b.toString(16).padStart(2, '0')).join('');
      const did = `did:deepfake:${randomId}`;
      const publicKey = `pk_${randomId.substring(0, 32)}`;

      const tx = await contract.registerDID(did, publicKey);
      await tx.wait();

      const didDoc = await contract.didDocuments(account);
      setUserDID(normalizeDidDocument(didDoc));
      alert("Đăng ký DID thành công!");
    } catch (error) {
      console.error("Register Error:", error);
      alert(formatTxError('đăng ký DID', error));
    } finally { setLoading(false); }
  };

  useEffect(() => {
    if (!window.ethereum) return;
    const handleChainChanged = (chainIdHex) => {
      const chainId = parseChainId(chainIdHex);
      if (Number.isFinite(chainId) && chainId !== EXPECTED_CHAIN_ID) {
        setContract(null);
        setUserDID(null);
        setVerificationResult(null);
        setStats({ totalDIDs: 0, totalVerifications: 0 });
      }
    };

    const handleAccountsChanged = (accounts) => {
      if (accounts.length > 0) {
        setAccount(accounts[0]);
        setUserDID(null);
        setVerificationResult(null);
      } else {
        setAccount(null);
        setContract(null);
        setUserDID(null);
        setVerificationResult(null);
      }
    };

    window.ethereum.on('chainChanged', handleChainChanged);
    window.ethereum.on('accountsChanged', handleAccountsChanged);

    return () => {
      window.ethereum.removeListener('chainChanged', handleChainChanged);
      window.ethereum.removeListener('accountsChanged', handleAccountsChanged);
    };
  }, []);

  const verifyImage = async (file) => {
    if (!file) return;
    if (!account) {
      alert("Vui lòng kết nối ví trước để xác thực!");
      return;
    }

    try {
      setLoading(true);
      setLoadingMessage('Đang phân tích ảnh bằng AI...');
      setVerificationResult(null);

      const formData = new FormData();
      formData.append('file', file);
      formData.append('user_address', account.toLowerCase());

      // 1. Call AI + ZKP API
      const response = await fetch(`${API_URL}/api/verify-zkp`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        let detail = 'Backend verification failed';
        try {
          const errorPayload = await response.json();
          detail = errorPayload?.detail || detail;
        } catch {
          // ignore json parse errors
        }
        throw new Error(detail);
      }
      const result = await response.json();

      const normalizedResult = {
        ...normalizeVerifyZkpResponse(result),
        onChain: false,
        pendingReason: null,
        onChainError: null
      };

      let finalResult = normalizedResult;

      // 2. Nếu ảnh REAL thì trích xuất dữ liệu ZKP và generate proof trên browser
      if (result.can_generate_proof && result.label === 'REAL' && result.zkp_input) {
        setLoadingMessage('Đang tạo bằng chứng Zero-Knowledge...');

        const oracle_secret = result.zkp_input.oracle_secret;
        const image_hash = result.image_hash;
        const confidence_int = Math.round((result.confidence || 0) * 10000);
        const timestamp = result.zkp_input.timestamp;

        const zkpPayload = {
          oracle_secret,
          image_hash,
          confidence_int,
          timestamp
        };

        const userSecretKey = `zkp_user_secret_${account.toLowerCase()}`;
        let userSecret = localStorage.getItem(userSecretKey);
        if (!userSecret) {
          const secretBytes = crypto.getRandomValues(new Uint8Array(16));
          const secretHex = Array.from(secretBytes).map(b => b.toString(16).padStart(2, '0')).join('');
          userSecret = BigInt(`0x${secretHex}`).toString();
          localStorage.setItem(userSecretKey, userSecret);
        }

        try {
          const browserZK = new BrowserZKProof();
          await browserZK.init();

          const proofResult = await browserZK.generateProof({
            label: 'REAL',
            image_hash,
            oracle_secret,
            timestamp,
            confidence_int
          }, userSecret);

          if (!proofResult.success) {
            throw new Error(proofResult.error || 'Không thể tạo ZK proof');
          }

          finalResult = {
            ...normalizedResult,
            zkp: {
              ...zkpPayload,
              commitment: proofResult.commitment,
              nullifier: proofResult.nullifier,
              proof: proofResult.proof,
              publicSignals: proofResult.publicSignals,
              calldata: proofResult.calldata
            }
          };
          alert('Đã tạo Zero-Knowledge proof thành công trên trình duyệt!');
        } catch (zkError) {
          console.error('ZKP generation failed:', zkError);
          finalResult = {
            ...normalizedResult,
            zkp: {
              ...zkpPayload,
              error: zkError.message
            }
          };
          alert('AI verify thành công nhưng tạo ZK proof thất bại: ' + zkError.message);
        }
      } else {
        finalResult = normalizedResult;
        if (result.label !== 'REAL') {
          alert('Ảnh được phân loại là FAKE nên không tạo Zero-Knowledge proof.');
        }
      }

      const onChainResult = await submitVerificationOnChain(result, finalResult);
      setVerificationResult(onChainResult);

      if (onChainResult.onChain) {
        alert('Đã ghi kết quả xác thực lên blockchain thành công.');
      }

    } catch (error) {
      console.error("Verification failed:", error);
      alert("Xác thực thất bại: " + error.message);
    } finally {
      setLoadingMessage('');
      setLoading(false);
    }
  };

  const fmtAddr = (a) => a ? `${a.substring(0, 6)}…${a.slice(-4)}` : ''

  return (
    <div className="app">
      <div className="bg-glow-1"></div>
      <div className="bg-glow-2"></div>

      {/* ── Header fixed ── */}
      <header className="header">
        <div className="header-content">
          {/* Logo trái */}
          <div className="logo brand-font">
            <div className="logo-icon">
              <Shield size={22} strokeWidth={2} />
            </div>
            <span className="logo-text">DeepTrust<span className="text-gradient">.AI</span></span>
          </div>

          {/* Wallet pill — phải */}
          <div className="header-nav">
            {account ? (
              <div className="wallet-pill">
                <span className="wallet-dot"></span>
                <span className="wallet-addr">{fmtAddr(account)}</span>
                <button
                  className="wallet-disconnect"
                  onClick={() => { setAccount(null); setContract(null); setUserDID(null); }}
                  title="Ngắt kết nối"
                >✕</button>
              </div>
            ) : (
              <button className="btn btn-primary header-connect-btn" onClick={connectWallet}>
                Kết nối ví
              </button>
            )}
          </div>
        </div>
      </header>

      {/* ── Section 1: Hero + Pillars — vừa 1 viewport ── */}
      <section className="section-hero">
        <div className="hero-body">
          <div className="page-header">
            <h1 className="page-title brand-font">
              Trust <span className="text-gradient">But Verify</span>
              <br /> in the Age of AI
            </h1>
            <p className="page-subtitle">
              Secure, decentralized authentication powered by artificial intelligence and zero-knowledge proofs.
            </p>
          </div>

          <div className="pillars-grid">
            <div className="pillar-card pillar-did">
              <div className="pillar-icon"><Shield size={24} strokeWidth={1.5} /></div>
              <h3 className="pillar-title">Decentralized Identity</h3>
              <p className="pillar-desc">
                Own your digital self. Cryptographic proof of personhood ensures you are in total control without central authorities.
              </p>
            </div>
            <div className="pillar-card pillar-ai">
              <div className="pillar-icon"><ScanFace size={24} strokeWidth={1.5} /></div>
              <h3 className="pillar-title">AI Deepfake Detection</h3>
              <p className="pillar-desc">
                State-of-the-art neural networks analyze micro-expressions to detect manipulated media with military-grade precision.
              </p>
            </div>
            <div className="pillar-card pillar-zkp">
              <div className="pillar-icon"><KeyRound size={24} strokeWidth={1.5} /></div>
              <h3 className="pillar-title">Zero-Knowledge Proofs</h3>
              <p className="pillar-desc">
                Verify authenticity mathematically without ever revealing your sensitive underlying biometric data.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ── Section 2: Action — cuộn xuống ── */}
      <section className="section-action">
        <div className="action-body">
          <div className="core-action-area">
            <div className="sidebar-action">
              <WalletConnect
                account={account} userDID={userDID}
                onConnect={connectWallet}
                onDisconnect={() => { setAccount(null); setContract(null); setUserDID(null); }}
                onRegisterDID={registerDID} loading={loading}
                stats={stats}
              />
            </div>

            <div className="main-action">
              <ImageUpload onUpload={verifyImage} loading={loading} />
              {loading && loadingMessage && (
                <p style={{ marginTop: '12px', color: 'var(--text-muted)' }}>{loadingMessage}</p>
              )}
              {verificationResult && (<VerificationResult result={verificationResult} />)}
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

export default App