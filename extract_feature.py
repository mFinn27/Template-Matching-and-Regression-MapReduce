import torch
import cv2
import numpy as np
import os
from torch.nn import functional as F

# --- IMPORT MODULE CÓ SẴN CỦA PROJECT ---
from models.backbone import build_backbone
from models.encoders import build_encoder
from utils.segment_anything.utils.transforms import ResizeLongestSide 

def run_extraction_and_analyze(image_path, output_dir="feature"):
    # ---------------------------------------------------------
    # 1. SETUP MODEL
    # ---------------------------------------------------------
    class Config:
        backbone = 'sam'
        encoder = 'original'
        emb_dim = 256
        dilation = False
    
    args = Config()
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[1/4] Đang khởi tạo model trên: {device}")

    try:
        backbone = build_backbone(args)
        model = build_encoder(args)(backbone, args.emb_dim)
        model.eval()
        model.to(device)
    except Exception as e:
        print(f"      -> LỖI Load Model: {e}")
        print("      -> Mẹo: Hãy chắc chắn bạn đã có file weights hoặc đã chỉnh code để bỏ qua load weight.")
        return

    # ---------------------------------------------------------
    # 2. XỬ LÝ ẢNH (PREPROCESSING)
    # ---------------------------------------------------------
    print(f"[2/4] Đang xử lý ảnh: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"      -> LỖI: Không tìm thấy file ảnh: {image_path}")
        return
        
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize & Normalize & Pad (Chuẩn SAM)
    target_size = 1024
    transform = ResizeLongestSide(target_size)
    input_image = transform.apply_image(image)
    
    input_image_torch = torch.as_tensor(input_image, device=device)
    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    
    pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(-1, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375], device=device).view(-1, 1, 1)
    x = (input_image_torch - pixel_mean) / pixel_std

    h, w = x.shape[-2:]
    x = F.pad(x, (0, target_size - w, 0, target_size - h))

    # ---------------------------------------------------------
    # 3. TRÍCH XUẤT & TÍNH TOÁN CHỈ SỐ (QUAN TRỌNG)
    # ---------------------------------------------------------
    print("[3/4] Đang chạy qua Backbone & Tính toán...")
    with torch.no_grad():
        features = model(x) # (1, 256, 64, 64)
    
    # Chuyển về Numpy để tính toán
    feat_np = features.cpu().numpy()
    
    # --- TÍNH CÁC CHỈ SỐ KHOA HỌC ---
    val_mean = np.mean(feat_np)
    val_std  = np.std(feat_np)
    val_max  = np.max(feat_np)
    val_sparsity = np.mean(feat_np <= 0) # Tỷ lệ phần tử <= 0

    # In kết quả ra Terminal đẹp mắt
    print("\n" + "="*60)
    print(f" KẾT QUẢ PHÂN TÍCH ĐẶC TRƯNG: {os.path.basename(image_path)}")
    print("="*60)
    print(f" 1. AVG ACTIVATION (Độ nhạy)   : {val_mean:.6f}")
    print(f" 2. STD (Độ tương phản)        : {val_std:.6f}")
    print(f" 3. MAX CONFIDENCE (Độ tự tin) : {val_max:.6f}")
    print(f" 4. SPARSITY (Độ thưa)         : {val_sparsity*100:.2f}%")
    print("-" * 60)
    
    # Đánh giá sơ bộ (Rule-based đơn giản)
    if val_mean < 0.0130:
        print(" => ĐÁNH GIÁ: Ảnh KHÓ (Hard) hoặc Ít thông tin.")
    elif val_mean > 0.0137:
        print(" => ĐÁNH GIÁ: Ảnh TỐT/BÌNH THƯỜNG (Normal/Easy).")
    else:
        print(" => ĐÁNH GIÁ: Trung bình.")
    print("="*60 + "\n")

    # ---------------------------------------------------------
    # 4. EXPORT FILE .NPY
    # ---------------------------------------------------------
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_name = os.path.basename(image_path).split('.')[0]
    save_path = os.path.join(output_dir, f"{base_name}_feature.npy")
    np.save(save_path, feat_np)
    print(f"[4/4] Đã lưu file đặc trưng tại: {save_path}")

if __name__ == "__main__":
    # --- CẤU HÌNH ĐƯỜNG DẪN ẢNH TẠI ĐÂY ---
    # Thay đổi thành đường dẫn ảnh bạn muốn test
    img_path = "demo/1.jpg" 
    
    # Tạo ảnh giả nếu không có file thật
    if not os.path.exists(img_path):
        print("Không thấy ảnh demo, tạo ảnh test tạm...")
        dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.imwrite("test_image.jpg", dummy_img)
        img_path = "test_image.jpg"

    run_extraction_and_analyze(img_path)