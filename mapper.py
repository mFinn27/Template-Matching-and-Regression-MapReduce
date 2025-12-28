import sys
import os
import tarfile
import numpy as np
import onnxruntime as ort
import subprocess
import shutil
from PIL import Image

# --- CẤU HÌNH ---
# Đường dẫn lệnh Hadoop trên Windows (hoặc Linux tùy môi trường)
HADOOP_CMD = "D:/hadoop-3.3.0/bin/hadoop.cmd"
HDFS_OUTPUT_DIR = "/user/hadoop/features_output"

def get_category(folder_name):
    """Xác định nhãn Easy/Normal/Hard từ tên thư mục"""
    if folder_name.startswith("Easy_"): return "Easy"
    if folder_name.startswith("Normal_"): return "Normal"
    if folder_name.startswith("Hard_"): return "Hard"
    return "Unknown"

def preprocess_image(img_path, input_shape=(1024, 1024)):
    """Đọc ảnh, resize và chuẩn hóa đầu vào cho Model"""
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(input_shape)
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        return None

def main():
    # 1. Load Model (Chỉ load 1 lần khi Mapper khởi động)
    if not os.path.exists("model.onnx"):
        sys.stderr.write("FATAL: model.onnx not found!\n")
        return

    try:
        # Tắt log của ONNX Runtime để đỡ rác
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        ort_session = ort.InferenceSession("model.onnx", sess_options)
        input_name = ort_session.get_inputs()[0].name
    except Exception as e:
        sys.stderr.write(f"FATAL: Model load failed: {e}\n")
        return

    # 2. Đọc danh sách file TAR từ stdin
    for line in sys.stdin:
        tar_filename = line.strip()
        if not tar_filename: continue

        folder_name = tar_filename.replace(".tar", "")
        category = get_category(folder_name)
        
        tar_sum_mean = 0.0
        tar_sum_std  = 0.0
        tar_sum_max  = 0.0
        tar_sum_spar = 0.0
        tar_image_count = 0

        # --- GIAI ĐOẠN 1: Tải và Giải nén ---
        try:
            # Clean up trước khi chạy
            if os.path.exists(folder_name): shutil.rmtree(folder_name, ignore_errors=True)
            
            # Tải từ HDFS
            hdfs_tar_path = f"/user/hadoop/data/tars/Gen_Tar_Data/{tar_filename}"
            subprocess.check_call([HADOOP_CMD, "fs", "-get", hdfs_tar_path, "."], stderr=subprocess.DEVNULL)
            
            # Giải nén
            with tarfile.open(tar_filename, "r") as tar:
                tar.extractall(path=folder_name)
            
            # Xóa file tar ngay để tiết kiệm chỗ
            os.remove(tar_filename)
        except Exception as e:
            sys.stderr.write(f"Error Setup {tar_filename}: {e}\n")
            continue

        # --- GIAI ĐOẠN 2: Inference & Tính Toán Chỉ Số ---
        output_local_folder = f"out_{folder_name}"
        os.makedirs(output_local_folder, exist_ok=True)
        
        for root, dirs, files in os.walk(folder_name):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, file)
                    
                    # Preprocess
                    img_tensor = preprocess_image(img_path)
                    if img_tensor is None: continue

                    # Inference
                    try:
                        feature = ort_session.run(None, {input_name: img_tensor})[0]
                        
                        # ====================================================
                        # TÍNH TOÁN 4 CHỈ SỐ KHOA HỌC
                        # ====================================================
                        val_mean = float(np.mean(feature))
                        val_std  = float(np.std(feature))
                        val_max  = float(np.max(feature))
                        # Sparsity: Tỷ lệ các phần tử <= 0 (hoặc xấp xỉ 0)
                        val_spar = float(np.mean(feature <= 0))

                        # Cộng dồn vào biến tổng của TAR
                        tar_sum_mean += val_mean
                        tar_sum_std  += val_std
                        tar_sum_max  += val_max
                        tar_sum_spar += val_spar
                        tar_image_count += 1
                        
                        # Lưu file NPY (Side-effect)
                        save_name = f"{os.path.splitext(file)[0]}.npy"
                        np.save(os.path.join(output_local_folder, save_name), feature)
                        
                    except Exception as e:
                        pass # Bỏ qua ảnh lỗi

        # --- GIAI ĐOẠN 3: Upload & Emit ---
        if tar_image_count > 0:
            # A. Upload file NPY lên HDFS
            final_hdfs_path = f"{HDFS_OUTPUT_DIR}/{category}/{folder_name}"
            try:
                subprocess.call([HADOOP_CMD, "fs", "-rm", "-r", final_hdfs_path], stderr=subprocess.DEVNULL)
                subprocess.call([HADOOP_CMD, "fs", "-mkdir", "-p", f"{HDFS_OUTPUT_DIR}/{category}"], stderr=subprocess.DEVNULL)
                subprocess.check_call([HADOOP_CMD, "fs", "-put", output_local_folder, final_hdfs_path], stderr=subprocess.DEVNULL)
                sys.stderr.write(f"Processed {tar_filename}: {tar_image_count} images\n")
            except Exception as e:
                sys.stderr.write(f"Upload Failed {tar_filename}: {e}\n")

            # B. EMIT DỮ LIỆU THỐNG KÊ (QUAN TRỌNG)
            # Format: Category TAB SumMean,SumStd,SumMax,SumSpar,Count
            # Gửi TỔNG (SUM) đi để Reducer tính trung bình sau
            print(f"{category}\t{tar_sum_mean},{tar_sum_std},{tar_sum_max},{tar_sum_spar},{tar_image_count}")

        # Cleanup Disk
        shutil.rmtree(folder_name, ignore_errors=True)
        shutil.rmtree(output_local_folder, ignore_errors=True)

if __name__ == "__main__":
    main()