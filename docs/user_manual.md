# User Manual (DeepfakeVerify)

Tài liệu này hướng dẫn sử dụng hệ thống cho 2 vai trò:

- Người dùng xác minh danh tính ở Main App
- Người dùng tham gia Consumer App bằng bằng chứng ZKP

## 1) Tổng quan luồng sử dụng

Luồng chuẩn:

1. Kết nối ví MetaMask.
2. Tải ảnh để backend AI kiểm tra REAL/FAKE.
3. Nhận kết quả xác minh + dữ liệu cần thiết.
4. (Tùy luồng) tạo và dùng bằng chứng ZKP.
5. Vào consumer portal để tiếp tục tác vụ (ví dụ voting) mà không lộ dữ liệu gốc.

## 2) Yêu cầu trước khi dùng

- Trình duyệt Chrome/Edge/Brave.
- Cài MetaMask extension.
- Đã chuyển sang đúng network (theo môi trường dự án).
- Có ETH testnet nếu thao tác blockchain cần gas.

## 3) Hướng dẫn Main App (`/`)

## 3.1 Kết nối ví

- Mở trang chủ ứng dụng.
- Nhấn `Connect MetaMask`.
- Xác nhận trong popup MetaMask.

Kết quả mong đợi:

- Hiển thị địa chỉ ví đã kết nối.
- Có thể thấy trạng thái DID (đã đăng ký/chưa đăng ký).

## 3.2 Đăng ký DID (nếu chưa có)

- Nhấn `Register Identity (Free)` ở khu vực DID.
- Chờ phản hồi thành công.

Kết quả mong đợi:

- DID được hiển thị.
- Có thể bấm `Copy DID`.

## 3.3 Xác minh ảnh

- Chọn ảnh chân dung rõ mặt.
- Gửi ảnh để hệ thống phân tích.

Kết quả có thể nhận:

- `REAL`: ảnh hợp lệ.
- `FAKE`: nghi ngờ deepfake.
- `ERROR` với `status != ok` (ví dụ không phát hiện mặt rõ).

Mẹo để tăng tỷ lệ xử lý thành công:

- Ảnh rõ nét, đủ sáng.
- Mặt nhìn chính diện.
- Tránh ảnh quá nhỏ, mờ, hoặc che khuất.

## 3.4 Khi cần dữ liệu ZKP

Tùy màn hình hiện tại, hệ thống có thể cho phép tải/nhận artifact phục vụ prove (ví dụ `proof.json`, `public.json`) để dùng ở consumer app.

## 4) Hướng dẫn Consumer App (`/consumer/`)

## 4.1 Vào trang consumer

- Truy cập đường dẫn `/consumer/`.
- Kết nối ví nếu được yêu cầu.

## 4.2 Verify Identity Page

- Tải lên 2 file:
  - `proof.json`
  - `public.json`
- Nhấn nút xác minh.

Nếu thành công:

- Hệ thống cho phép đi tiếp sang chọn voting.

Nếu thất bại:

- Xem thông báo lỗi để kiểm tra lại định dạng file hoặc dữ liệu proof.

## 4.3 Select Voting Page

- Xem danh sách cuộc bầu chọn.
- Chọn một voting đang mở.
- Vào màn hình vote để thực hiện bỏ phiếu.

## 5) Giải thích nhanh các trạng thái

- `Connected Wallet`: ví đã kết nối.
- `DID Active`: danh tính phi tập trung đã đăng ký.
- `status=ok`: kết quả detector có thể dùng cho phân loại.
- `status!=ok`: kết quả non-classifiable, không dùng để approve.

## 6) Lỗi thường gặp và cách xử lý

### Không kết nối được MetaMask

- Kiểm tra extension đã mở quyền cho site.
- Kiểm tra đúng network chain id.
- Refresh trang và thử lại.

### Xác minh ảnh báo lỗi không phát hiện khuôn mặt

- Đổi ảnh rõ hơn, nhìn thẳng, đủ sáng.
- Tránh ảnh có nhiều người hoặc khuôn mặt quá nhỏ.

### Upload proof/public thất bại ở consumer

- Đảm bảo đúng cặp file cùng một phiên tạo proof.
- Kiểm tra file JSON không bị sửa thủ công.

### Không thấy voting

- Kiểm tra ví đã connect.
- Kiểm tra backend và blockchain đang hoạt động.

## 7) Quy tắc an toàn cho người dùng

- Không chia sẻ private key hoặc seed phrase.
- Chỉ ký các giao dịch/message bạn hiểu rõ.
- Kiểm tra đúng domain chính thức trước khi thao tác.

## 8) FAQ ngắn

### Hệ thống có lưu ảnh gốc của tôi không?

Tùy theo chính sách triển khai, nhưng luồng thiết kế hướng tới giảm lộ dữ liệu thô và dùng bằng chứng để xác minh thay vì chia sẻ thông tin nhạy cảm.

### Tại sao ảnh REAL vẫn không qua?

Có thể do chất lượng ảnh, lỗi detect khuôn mặt, hoặc lỗi kỹ thuật tạm thời (`status != ok`).

### Có bắt buộc dùng DID không?

Một số chức năng nâng cao (liên quan xác thực danh tính/credential) yêu cầu DID để đảm bảo tính định danh phi tập trung.

## 9) Checklist tự kiểm tra trước khi báo lỗi cho admin

- Đã kết nối ví thành công.
- Đúng network.
- Ảnh đầu vào rõ và hợp lệ.
- Backend `/api/health` phản hồi bình thường.
- Đã thử lại bằng trình duyệt khác.

Nếu vẫn lỗi, gửi kèm:

- Thời điểm lỗi
- Ảnh chụp màn hình
- Nội dung lỗi hiển thị
- Loại thao tác đang thực hiện (`verify`, `verify-zkp`, `consumer verify`, ...)
