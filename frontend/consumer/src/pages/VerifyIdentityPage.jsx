import { useContext, useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { VotingContext } from "../context/VotingContext";

const VerifyIdentityPage = () => {
  const { currentAccount, zkpVerified, verifyIdentity, isLoading } = useContext(VotingContext);
  const navigate = useNavigate();

  const [proofFile, setProofFile] = useState(null);
  const [publicFile, setPublicFile] = useState(null);
  const [result, setResult] = useState(null);
  const proofRef = useRef();
  const publicRef = useRef();

  if (!currentAccount) {
    return (
      <div className="page-message">
        <h2>Chưa kết nối ví</h2>
        <button className="btn btn-outline" onClick={() => navigate("/")}>Quay lại</button>
      </div>
    );
  }

  const handleProofChange = (e) => {
    const file = e.target.files[0];
    if (file) setProofFile(file);
    setResult(null);
  };

  const handlePublicChange = (e) => {
    const file = e.target.files[0];
    if (file) setPublicFile(file);
    setResult(null);
  };

  const handleVerify = async () => {
    if (!proofFile || !publicFile) return alert("Vui lòng chọn cả file proof.json và public.json!");
    try {
      const proofText = await proofFile.text();
      const publicText = await publicFile.text();
      const proof = JSON.parse(proofText);
      const publicSignals = JSON.parse(publicText);

      const res = await verifyIdentity(proof, publicSignals);
      setResult(res);
    } catch {
      setResult({ success: false, message: "Lỗi đọc file JSON: Cần file JSON hợp lệ của snarkjs." });
    }
  };

  return (
    <div className="verify-identity-page">
      <h2>Xác Minh Danh Tính Ẩn Danh</h2>
      <p className="desc">
        Là một Provider Dịch Vụ, chúng tôi không chụp hay xử lý ảnh khuôn mặt của bạn. Yêu cầu tạo bằng chứng ZKP Toán Học từ App Chính (Identity Provider) và tải lên đây.
      </p>

      {zkpVerified ? (
        <div className="success-box">
          <h3>Xác minh Cấu trúc Toán học thành công!</h3>
          <p>
            Khớp khóa giải mã. Bạn đã chứng minh danh tính thật của mình mà không lộ dữ liệu thô!
          </p>
          <button
            className="btn btn-primary"
            onClick={() => navigate("/select-voting")}
          >
            Tham gia Bỏ Phiếu
          </button>
        </div>
      ) : (
        <div className="verify-form">
          <div className="verify-upload-grid">
            <div
              className={`upload-area ${proofFile ? 'uploaded' : ''}`}
              onClick={() => proofRef.current?.click()}
            >
              <h4 style={{ margin: "0 0 10px 0" }}>Tải lên Proof</h4>
              <p className="upload-placeholder">{proofFile ? proofFile.name : "Chọn file proof.json"}</p>
              <input ref={proofRef} type="file" accept=".json" onChange={handleProofChange} hidden />
            </div>

            <div
              className={`upload-area ${publicFile ? 'uploaded' : ''}`}
              onClick={() => publicRef.current?.click()}
            >
              <h4 style={{ margin: "0 0 10px 0" }}>Tải lên Public Signals</h4>
              <p className="upload-placeholder">{publicFile ? publicFile.name : "Chọn file public.json"}</p>
              <input ref={publicRef} type="file" accept=".json" onChange={handlePublicChange} hidden />
            </div>
          </div>

          <button className="btn btn-primary btn-large" onClick={handleVerify} disabled={!proofFile || !publicFile || isLoading} style={{ width: "100%" }}>
            {isLoading ? "Đang xác minh..." : "Xác Minh Bằng Chứng Toán Học"}
          </button>

          {result && !result.success && (
            <div className="error-box">
              <strong>Xác minh thất bại:</strong> {result.message}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default VerifyIdentityPage;

