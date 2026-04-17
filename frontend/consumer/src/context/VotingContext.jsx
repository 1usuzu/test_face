/* eslint-disable react-refresh/only-export-components */
import React, { createContext, useState, useEffect } from "react";
import { BrowserProvider, Contract, isAddress } from "ethers";

import VotingFactoryABI from "../contracts/VotingFactory.json";
import VotingABI from "../contracts/Voting.json";

export const VotingContext = createContext();

const EXPECTED_CHAIN_ID = Number(import.meta.env.VITE_CHAIN_ID || 31337);
const factoryAddress = import.meta.env.VITE_VOTING_FACTORY_ADDRESS || VotingFactoryABI.networks?.["31337"]?.address;
const factoryABI = VotingFactoryABI.abi || [];
const votingABI = VotingABI.abi || [];

export const VotingProvider = ({ children }) => {
  // ── Wallet ────────────────────────────────────────────────────────────────
  const [currentAccount, setCurrentAccount] = useState(null);
  const [factoryContract, setFactoryContract] = useState(null);

  // ── Voting ────────────────────────────────────────────────────────────────
  const [currentVotingContract, setCurrentVotingContract] = useState(null);
  const [currentVotingAddress, setCurrentVotingAddress] = useState(null);
  const [candidates, setCandidates] = useState([]);
  const [isAdmin, setIsAdmin] = useState(false);
  const [votingOpen, setVotingOpen] = useState(true);
  const [votingInfo, setVotingInfo] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  // ── ZKP Identity ─────────────────────────────────────────────────────────
  const [zkpVerified, setZkpVerified] = useState(false);
  const [zkpData, setZkpData] = useState(null); // raw response từ Core API

  // ── Provider / Signer ─────────────────────────────────────────────────────
  const getProviderAndSigner = async () => {
    if (!window.ethereum) return null;
    try {
      const provider = new BrowserProvider(window.ethereum);
      const network = await provider.getNetwork();
      if (Number(network.chainId) !== EXPECTED_CHAIN_ID) {
        try {
          await window.ethereum.request({
            method: "wallet_switchEthereumChain",
            params: [{ chainId: `0x${EXPECTED_CHAIN_ID.toString(16)}` }],
          });
          window.location.reload();
        } catch {
          return null;
        }
        return null;
      }
      const signer = await provider.getSigner();
      return { provider, signer };
    } catch {
      return null;
    }
  };

  // ── Factory Contract ──────────────────────────────────────────────────────
  const initFactoryContract = async () => {
    if (!factoryAddress) return null;
    const result = await getProviderAndSigner();
    if (!result) return null;
    try {
      const factory = new Contract(factoryAddress, factoryABI, result.signer);
      setFactoryContract(factory);
      return factory;
    } catch {
      return null;
    }
  };

  // ── Create Voting ─────────────────────────────────────────────────────────
  const createVoting = async (title, description) => {
    if (!factoryAddress) {
      alert("VotingFactory chưa được deploy!");
      return false;
    }
    try {
      setIsLoading(true);
      const contract = factoryContract || await initFactoryContract();
      if (!contract) { alert("Không thể kết nối với contract!"); return false; }

      const tx = await contract.createVoting(title, description);
      const receipt = await tx.wait();

      const event = receipt.logs.find(log => {
        try { return contract.interface.parseLog(log)?.name === "VotingCreated"; }
        catch { return false; }
      });
      if (event) {
        const parsed = contract.interface.parseLog(event);
        await selectVoting(parsed.args.votingAddress);
      }
      return true;
    } catch (error) {
      alert(error.reason || error.message);
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  // ── Get All Votings ───────────────────────────────────────────────────────
  const getAllVotings = async () => {
    if (!factoryAddress) return [];
    try {
      const contract = factoryContract || await initFactoryContract();
      if (!contract) return [];

      const addresses = await contract.getAllVotings();
      const result = await getProviderAndSigner();
      if (!result) return [];

      const list = await Promise.all(
        addresses.map(async (addr) => {
          try {
            const c = new Contract(addr, votingABI, result.signer);
            const info = await c.getVotingInfo();
            return {
              address: addr,
              title: info.title,
              description: info.desc,
              admin: info.adminAddr,
              candidateCount: Number(info.candidateCount),
              isOpen: info.isOpen,
              createdAt: new Date(Number(info.created) * 1000).toLocaleString("vi-VN"),
              isAdmin: currentAccount && info.adminAddr.toLowerCase() === currentAccount.toLowerCase(),
            };
          } catch { return null; }
        })
      );
      return list.filter(Boolean);
    } catch {
      return [];
    }
  };

  // ── Select Voting ─────────────────────────────────────────────────────────
  const selectVoting = async (votingAddress) => {
    const result = await getProviderAndSigner();
    if (!result) { alert("Không thể kết nối ví!"); return false; }
    try {
      const votingContract = new Contract(votingAddress, votingABI, result.signer);
      setCurrentVotingContract(votingContract);
      setCurrentVotingAddress(votingAddress);
      await loadVotingData(votingContract);
      return true;
    } catch {
      alert("Không thể kết nối với cuộc bầu chọn này!");
      return false;
    }
  };

  // ── Load Voting Data ──────────────────────────────────────────────────────
  const loadVotingData = async (contract = currentVotingContract) => {
    if (!contract) return;
    try {
      const info = await contract.getVotingInfo();
      setVotingInfo({
        title: info.title,
        description: info.desc,
        admin: info.adminAddr,
        candidateCount: Number(info.candidateCount),
        isOpen: info.isOpen,
        createdAt: new Date(Number(info.created) * 1000).toLocaleString("vi-VN"),
      });
      setVotingOpen(info.isOpen);
      if (currentAccount) {
        setIsAdmin(currentAccount.toLowerCase() === info.adminAddr.toLowerCase());
      }
      if (Number(info.candidateCount) > 0) {
        const all = await contract.getAllCandidates();
        setCandidates(all.map(c => ({
          id: Number(c.id),
          name: c.name,
          candidateAddress: c.candidateAddress,
          voteCount: Number(c.voteCount),
        })));
      } else {
        setCandidates([]);
      }
    } catch (error) {
      console.error("Error loading voting data:", error);
    }
  };

  // ── Vote ──────────────────────────────────────────────────────────────────
  const vote = async (candidateId) => {
    if (!currentVotingContract) { alert("Chưa chọn cuộc bầu chọn!"); return; }
    try {
      setIsLoading(true);
      const tx = await currentVotingContract.vote(candidateId);
      await tx.wait();
      await loadVotingData();
      alert("Bỏ phiếu thành công!");
    } catch (error) {
      alert(error.reason || error.message);
    } finally {
      setIsLoading(false);
    }
  };

  // ── Add Candidate ─────────────────────────────────────────────────────────
  const addCandidate = async (name, address) => {
    if (!currentVotingContract) { alert("Chưa chọn cuộc bầu chọn!"); return; }
    if (!isAddress(address)) { alert("Địa chỉ ví ứng viên không hợp lệ!"); return; }
    try {
      setIsLoading(true);
      const tx = await currentVotingContract.addCandidate(name, address);
      await tx.wait();
      await loadVotingData();
      alert("Thêm ứng viên thành công!");
    } catch (error) {
      alert(error.reason || error.message);
    } finally {
      setIsLoading(false);
    }
  };

  // ── Toggle Voting ─────────────────────────────────────────────────────────
  const closeVoting = async () => {
    if (!currentVotingContract) return;
    try {
      setIsLoading(true);
      const tx = await currentVotingContract.closeVoting();
      await tx.wait();
      await loadVotingData();
      alert("Đã đóng bầu cử thành công!");
    } catch (error) {
      alert(error.reason || error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const openVoting = async () => {
    if (!currentVotingContract) return;
    try {
      setIsLoading(true);
      const tx = await currentVotingContract.openVoting();
      await tx.wait();
      await loadVotingData();
      alert("Đã mở bầu cử thành công!");
    } catch (error) {
      alert(error.reason || error.message);
    } finally {
      setIsLoading(false);
    }
  };

  // ── ZKP Identity Verify ───────────────────────────────────────────────────
  const verifyIdentity = async (proof, publicSignals) => {
    if (!currentAccount) return { success: false, message: "Chưa kết nối ví" };

    // Front-end binding check: Mặc dù Zero-Knowledge Proof bảo vệ danh tính,
    // ta chốt thêm lớp bảo mật: Yêu cầu file proof phải được sinh ra trên chính trình duyệt cùng loại Ví đã ký.
    const userSecret = localStorage.getItem(`zkp_user_secret_${currentAccount}`);
    if (!userSecret) {
      return {
        success: false,
        message: "Bằng chứng ZKP không khớp với tài khoản ví hiện tại! Hãy đảm bảo bạn dùng đúng tài khoản MetaMask đã nhận chứng nhận trước đó."
      };
    }

    try {
      setIsLoading(true);

      if (typeof window.snarkjs === "undefined") {
        return { success: false, message: "Thư viện snarkjs chưa sẵn sàng!" };
      }

      // Fetch the verification key
      const vKeyRes = await fetch("/verification_key.json?t=" + Date.now());
      if (!vKeyRes.ok) {
        return { success: false, message: "Không tìm thấy verification_key.json" };
      }
      const vKey = await vKeyRes.json();

      // Verify off-chain
      const res = await window.snarkjs.groth16.verify(vKey, publicSignals, proof);

      if (res === true) {
        setZkpVerified(true);
        setZkpData({ proof, publicSignals });
        return { success: true, message: "Xác minh Bằng chứng hợp lệ!" };
      } else {
        return { success: false, message: "Bằng chứng ZKP không hợp lệ (FAKE)!" };
      }
    } catch (error) {
      console.error("ZKP Verification Error:", error);
      return { success: false, message: "Lỗi giải mã: " + error.message };
    } finally {
      setIsLoading(false);
    }
  };

  // ── Wallet Connect ────────────────────────────────────────────────────────
  const connectWallet = async () => {
    if (!window.ethereum) { alert("Vui lòng cài đặt MetaMask!"); return; }
    try {
      const accounts = await window.ethereum.request({ method: "eth_requestAccounts" });
      setCurrentAccount(accounts[0].toLowerCase());
      await initFactoryContract();
    } catch (error) {
      console.error("Error connecting wallet:", error);
    }
  };

  const disconnectWallet = () => {
    setCurrentAccount(null);
    setFactoryContract(null);
    setCurrentVotingContract(null);
    setCurrentVotingAddress(null);
    setCandidates([]);
    setIsAdmin(false);
    setVotingInfo(null);
    setZkpVerified(false);
    setZkpData(null);
  };

  // ── Init ──────────────────────────────────────────────────────────────────
  useEffect(() => {
    const init = async () => {
      if (!window.ethereum) return;
      const accounts = await window.ethereum.request({ method: "eth_accounts" });
      if (accounts.length > 0) {
        setCurrentAccount(accounts[0].toLowerCase());
        await initFactoryContract();
      }
      window.ethereum.on("accountsChanged", (accounts) => {
        if (accounts.length > 0) {
          const newAccount = accounts[0].toLowerCase();
          setCurrentAccount(newAccount);
          setZkpVerified(false);
          setZkpData(null);
          initFactoryContract();
        } else {
          disconnectWallet();
        }
      });
      window.ethereum.on("chainChanged", () => window.location.reload());
    };
    init();
    return () => { window.ethereum?.removeAllListeners?.(); };
  }, []);

  useEffect(() => {
    if (currentVotingContract && currentAccount) loadVotingData();
  }, [currentAccount, currentVotingContract]);

  return (
    <VotingContext.Provider value={{
      currentAccount, connectWallet, disconnectWallet,
      createVoting, getAllVotings,
      selectVoting, currentVotingAddress, votingInfo,
      candidates, isLoading, isAdmin, votingOpen,
      vote, addCandidate, closeVoting, openVoting,
      zkpVerified, zkpData, verifyIdentity,
    }}>
      {children}
    </VotingContext.Provider>
  );
};

