"""
verify_upgrade.py - Script chứng minh hiệu quả của Face Extraction
"""
import torch
from detect import DeepfakeDetector
from facenet_pytorch import MTCNN
import sys
import os

# --- CẤU HÌNH ---
# Hãy thay bằng đường dẫn tới 1 bức ảnh KHÓ (ví dụ: ảnh người đứng xa, hoặc ảnh có nền phức tạp)
TEST_IMAGE_PATH = "test_image.jpg" 

def run_test():
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ Lỗi: Không tìm thấy file ảnh '{TEST_IMAGE_PATH}'")
        print("   -> Hãy chép 1 bức ảnh bất kỳ vào đây và đổi tên thành 'test_image.jpg'")
        return

    print("="*60)
    print(f"🧪 KIỂM TRA HIỆU NĂNG: v2 (Cũ) vs v3 (Mới)")
    print(f"🖼️  Ảnh test: {TEST_IMAGE_PATH}")
    print("="*60)

    # Khởi tạo Detector
    detector = DeepfakeDetector()
    
    # ---------------------------------------------------------
    # TEST CASE 1: GIẢ LẬP MODEL CŨ (Không có MTCNN)
    # ---------------------------------------------------------
    print("\n🔻 CASE 1: Chạy kiểu CŨ (Resize toàn bộ ảnh -> Model)")
    
    # Tạm thời tắt MTCNN đi để giả lập code cũ
    real_mtcnn = detector.mtcnn
    detector.mtcnn = None 
    
    result_v2 = detector.predict(TEST_IMAGE_PATH)
    print(f"   👉 Kết quả: {'FAKE' if result_v2.is_fake else 'REAL'}")
    print(f"   📉 Điểm số (Confidence): {result_v2.fake_probability:.4f}")
    print(f"   ℹ️  Chi tiết: {result_v2.details}")

    # ---------------------------------------------------------
    # TEST CASE 2: MODEL MỚI (Có MTCNN - Face Extraction)
    # ---------------------------------------------------------
    print("\n🔺 CASE 2: Chạy kiểu MỚI (Cắt mặt MTCNN -> Model)")
    
    # Bật lại MTCNN
    detector.mtcnn = real_mtcnn
    
    if detector.mtcnn is None:
        print("⚠️  Lỗi: Không load được MTCNN. Bạn đã cài 'facenet-pytorch' chưa?")
        return

    result_v3 = detector.predict(TEST_IMAGE_PATH)
    print(f"   👉 Kết quả: {'FAKE' if result_v3.is_fake else 'REAL'}")
    print(f"   📈 Điểm số (Confidence): {result_v3.fake_probability:.4f}")
    print(f"   ℹ️  Chi tiết: {result_v3.details}")

    # ---------------------------------------------------------
    # ĐÁNH GIÁ
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("KẾT LUẬN")
    print("="*60)
    
    diff = abs(result_v3.fake_probability - result_v2.fake_probability)
    
    if result_v3.details.get('face_detected'):
        print(f"✅ Đã tìm thấy khuôn mặt!")
        print(f"📊 Độ lệch điểm số: {diff:.4f}")
        
        if result_v3.fake_probability > result_v2.fake_probability:
            print("🚀 Phiên bản MỚI phát hiện dấu hiệu giả mạo RÕ RÀNG HƠN.")
        else:
            print("ℹ️  Hai phiên bản cho kết quả tương đồng (Ảnh này quá dễ hoặc quá khó).")
    else:
        print("⚠️  Không tìm thấy mặt trong ảnh (MTCNN thất bại).")
        print("   -> Code tự động fallback về cách cũ nên kết quả giống nhau.")

if __name__ == "__main__":
    run_test()