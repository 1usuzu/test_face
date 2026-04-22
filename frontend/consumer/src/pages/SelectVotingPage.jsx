import { useContext, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { VotingContext } from "../context/VotingContext";

const SelectVotingPage = () => {
  const { currentAccount, getAllVotings, selectVoting } = useContext(VotingContext);
  const navigate = useNavigate();
  const [votings, setVotings] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      setLoading(true);
      const list = await getAllVotings();
      setVotings(list || []);
      setLoading(false);
    })();
  }, [currentAccount]);

  const handleSelect = async (addr) => {
    const ok = await selectVoting(addr);
    if (ok) navigate("/vote");
  };

  if (!currentAccount) {
    return (
      <div className="page-message">
        <h2>Chưa kết nối ví</h2>
        <p>Vui lòng kết nối MetaMask để xem danh sách bầu chọn.</p>
        <button className="btn btn-outline" onClick={() => navigate("/")}>Quay lại</button>
      </div>
    );
  }

  return (
    <div className="select-voting-page">
      <h2>Chọn Cuộc Bầu Chọn</h2>
      <p className="desc">Chọn một cuộc bầu chọn để tham gia bỏ phiếu ẩn danh</p>

      {loading ? (
        <div className="loading">Đang tải danh sách bầu chọn...</div>
      ) : votings.length === 0 ? (
        <div className="empty-state">
          <p>Chưa có cuộc bầu chọn nào</p>
          <button className="btn btn-primary" onClick={() => navigate("/create-voting")}>
            Tạo cuộc bầu chọn đầu tiên
          </button>
        </div>
      ) : (
        <div className="voting-grid">
          {votings.map((v, i) => (
            <div key={i} className="voting-card" onClick={() => handleSelect(v.address)}>
              <h3>{v.title}</h3>
              {v.description && <p className="voting-desc">{v.description}</p>}
              <div className="voting-meta">
                <p><strong>Số ứng viên:</strong> {v.candidateCount}</p>
                <p>
                  <strong>Trạng thái:</strong>{" "}
                  <span className={v.isOpen ? "status-open" : "status-closed"}>
                    {v.isOpen ? "Đang mở" : "Đã đóng"}
                  </span>
                </p>
                <p>
                  <strong>Admin:</strong>{" "}
                  <span className={v.isAdmin ? "is-you" : ""}>
                    {v.isAdmin ? "Bạn" : `${v.admin.slice(0, 6)}...${v.admin.slice(-4)}`}
                  </span>
                </p>
                <p className="created-at">{v.createdAt}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="page-footer">
        <button className="btn btn-outline" onClick={() => navigate("/")}>Quay lại trang chủ</button>
      </div>
    </div>
  );
};

export default SelectVotingPage;
