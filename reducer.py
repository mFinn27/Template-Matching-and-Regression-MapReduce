import sys
import numpy as np

def process_batch_and_print(category, stats_list):
    """Tính toán thống kê từ dữ liệu đã tổng hợp"""
    if not stats_list:
        sys.stderr.write(f"[WARNING] No stats for category: {category}\n")
        return

    try:
        # Tổng hợp từ các TAR
        total_images = sum(s['count'] for s in stats_list)
        total_mean = sum(s['sum_mean'] for s in stats_list)
        total_std = sum(s['sum_std'] for s in stats_list)
        total_max = sum(s['sum_max'] for s in stats_list)
        total_spar = sum(s['sum_spar'] for s in stats_list)
        
        # Tính trung bình
        avg_mean = total_mean / total_images
        avg_std = total_std / total_images
        avg_max = total_max / total_images
        avg_spar = total_spar / total_images
        
        # In kết quả (format đơn giản hơn vì không có raw data)
        print(f"{category:<12} | {total_images:>6} | "
            f"{avg_mean:>8.4f} | {avg_std:>8.4f} | "
            f"{avg_max:>8.4f} | {avg_spar:>7.2%}")
        
        sys.stderr.write(f"[INFO] Completed {category}: {total_images} images from {len(stats_list)} TARs\n")
        
    except Exception as e:
        sys.stderr.write(f"[ERROR] Failed to calculate stats for {category}: {e}\n")

def main():
    current_category = None
    batch_stats = []
    
    # Header đơn giản (vì không có raw data để tính percentiles, L2 norm, etc.)
    print(f"{'CATEGORY':<12} | {'IMAGES':>6} | "
        f"{'AVG_MEAN':>8} | {'AVG_STD':>8} | "
        f"{'AVG_MAX':>8} | {'SPARSITY':>9}")
    print("-" * 70)
    
    sys.stderr.write("[INFO] Reducer started\n")
    line_count = 0

    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        
        line_count += 1
        parts = line.split('\t')
        if len(parts) != 2:
            sys.stderr.write(f"[WARNING] Invalid line format: {line}\n")
            continue
        
        category = parts[0]
        stats_str = parts[1]  # "0.234,0.421,1.234,0.42,100"
        
        # Parse thống kê
        try:
            values = stats_str.split(',')
            if len(values) != 5:
                sys.stderr.write(f"[WARNING] Invalid stats format: {stats_str}\n")
                continue
            
            stats = {
                'sum_mean': float(values[0]),
                'sum_std': float(values[1]),
                'sum_max': float(values[2]),
                'sum_spar': float(values[3]),
                'count': int(values[4])
            }
        except Exception as e:
            sys.stderr.write(f"[ERROR] Failed to parse stats: {stats_str} - {e}\n")
            continue
        
        # Logic gom nhóm (Aggregation)
        if current_category and category != current_category:
            process_batch_and_print(current_category, batch_stats)
            batch_stats = []

        current_category = category
        batch_stats.append(stats)
        
        # Log tiến trình mỗi 100 dòng
        if line_count % 100 == 0:
            sys.stderr.write(f"[PROGRESS] Processed {line_count} lines\n")

    # Xử lý nhóm cuối cùng
    if current_category and batch_stats:
        process_batch_and_print(current_category, batch_stats)
    
    sys.stderr.write(f"[INFO] Reducer finished. Total lines: {line_count}\n")

if __name__ == "__main__":
    main()