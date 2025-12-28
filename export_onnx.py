import sys
import os
import subprocess
import torch
import numpy as np

try:
    import onnx
except ImportError:
    print("⚠️ Phát hiện thiếu thư viện 'onnx'. Đang tự động cài đặt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx"])
    import onnx

sys.path.insert(0, "Package_Modules.zip") 
from models.backbone.sam.sam import Sam_Backbone 

def export_model():
    print("--- BẮT ĐẦU QUÁ TRÌNH EXPORT ONNX (ViT-B) ---")
    output_dir = "onnx_model_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device("cpu")
    
    print("1. Đang khởi tạo kiến trúc SAM ViT-B (Base)...")
    try:
        model = Sam_Backbone(requires_grad=False, model_path=None, model_type="vit_b")
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"❌ Lỗi khởi tạo model: {e}")
        return

    pth_path = "checkpoints/sam_hq_vit_b.pth"
    
    if os.path.exists(pth_path):
        print(f"2. Load trọng số từ {pth_path}...")
        try:
            checkpoint = torch.load(pth_path, map_location="cpu")
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("image_encoder."):
                    new_state_dict[k.replace("image_encoder.", "backbone.")] = v
                elif not k.startswith("backbone.") and "neck" not in k:
                    new_state_dict[f"backbone.{k}"] = v
                else:
                    new_state_dict[k] = v
            
            try:
                model.load_state_dict(new_state_dict, strict=False)
                print("-> Load trọng số thành công (Strict=False).")
            except RuntimeError as e:
                print("-> Thử load trực tiếp không qua xử lý key...")
                model.load_state_dict(state_dict, strict=False)
                
        except Exception as e:
            print(f"❌ Lỗi load weight: {e}")
            return
    else:
        print(f"❌ LỖI: Không tìm thấy file '{pth_path}'")
        return

    output_onnx = os.path.join(output_dir, "sam_vit_b_encoder.onnx")
    
    print("3. Đang tạo dummy input (1, 3, 1024, 1024)...")
    dummy_input = torch.randn(1, 3, 1024, 1024, device=device)

    print(f"4. Đang export vào '{output_onnx}'... (Chờ 1-2 phút)")
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_onnx,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input_image'],
            output_names=['image_embeddings'],
            dynamic_axes={
                'input_image': {0: 'batch_size'},
                'image_embeddings': {0: 'batch_size'}
            }
        )
        print(f"\n✅ THÀNH CÔNG TUYỆT ĐỐI! File nằm tại: {output_onnx}")
        file_size = os.path.getsize(output_onnx) / (1024*1024)
        print(f"Kích thước file: {file_size:.2f} MB")
        
        if file_size < 1800:
            print("-> File nhỏ hơn 2GB nên sẽ KHÔNG CÓ file ngoại lai đi kèm. Rất gọn gàng!")
        
    except Exception as e:
        print(f"\n❌ Lỗi khi export: {e}")

if __name__ == "__main__":
    export_model()