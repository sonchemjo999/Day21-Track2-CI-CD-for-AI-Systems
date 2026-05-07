# Báo Cáo Lab MLOps

**Họ và tên:** Đào Hồng Sơn  
**MSSV:** 2A202600462  
**Môn học:** AIInAction - Day 21 CI/CD cho AI Systems

## 1. Kết Quả Bước 1 - MLflow Tracking

Em đã chạy nhiều thí nghiệm huấn luyện mô hình `RandomForestClassifier` và theo dõi bằng MLflow. Bộ siêu tham số được chọn là:

```yaml
n_estimators: 450
max_depth: 20
min_samples_split: 2
```

Kết quả tốt nhất trên tập `data/eval.csv`:

| Chỉ số | Giá trị |
|---|---:|
| accuracy | 0.756 |
| f1_score | 0.7551 |

Bộ tham số này được chọn vì cho kết quả `accuracy` và `f1_score` cao nhất trong các lần thử nghiệm đã ghi nhận trên MLflow. Các ảnh bằng chứng nằm trong `screenshots/buoc1/`.

## 2. Kết Quả Bước 2 - CI/CD, DVC Và Serving

Dữ liệu đã được quản lý bằng DVC và đẩy lên AWS S3 bucket:

```text
s3://mlops-lab-son-511590151379/dvc/
```

Model sau huấn luyện được upload lên:

```text
s3://mlops-lab-son-511590151379/models/latest/model.pkl
```

GitHub Actions pipeline đã chạy thành công các job:

```text
Test -> Train -> Eval -> Deploy
```

Eval gate kiểm tra `accuracy >= 0.70` và run hiện tại đạt:

```text
PASSED: accuracy 0.7560 >= 0.7
```

Mô hình được triển khai bằng FastAPI trên EC2. Endpoint kiểm tra:

```text
GET /health  -> {"status":"ok"}
POST /predict -> {"prediction":2,"label":"cao"}
```

Các ảnh bằng chứng nằm trong `screenshots/buoc2/`.

## 3. Kết Quả Bước 3 - Huấn Luyện Liên Tục

Tập huấn luyện đã được bổ sung dữ liệu mới từ `train_phase2.csv`, làm số mẫu trong `train_phase1.csv` tăng lên:

```text
5996 mẫu
```

File DVC pointer liên quan đã được commit:

```text
data/train_phase1.csv.dvc
data/eval.csv.dvc
data/train_phase2.csv.dvc
```

DVC báo trạng thái:

```text
Data and pipelines are up to date.
```

Pipeline GitHub Actions sau đó chạy lại thành công trên trạng thái dữ liệu mới. Các ảnh bằng chứng nằm trong `screenshots/buoc3/`.

## 4. Bonus Đã Thực Hiện

Em đã thực hiện thêm 2 mục bonus:

- **Bonus 3:** Tự động tạo báo cáo hiệu suất `outputs/report.txt`, gồm confusion matrix, precision, recall và f1-score cho từng lớp.
- **Bonus 5:** Ghi phân phối nhãn huấn luyện vào `outputs/metrics.json` dưới khóa `label_distribution`.

Phân phối nhãn hiện tại:

```json
{
  "0": 0.3686,
  "1": 0.4351,
  "2": 0.1963
}
```

Các ảnh bằng chứng bonus nằm trong `screenshots/bonus/`.

## 5. Khó Khăn Và Cách Giải Quyết

Một số khó khăn gặp phải trong quá trình làm lab:

- IAM user AWS ban đầu thiếu quyền truy cập S3, đã bổ sung quyền `ListBucket`, `GetObject`, `PutObject`, `DeleteObject` cho bucket lab.
- DVC trên Windows gặp lỗi cache database, đã cấu hình `core.site_cache_dir = .dvc/site-cache`.
- File SSH key `.pem` trên Windows bị lỗi permission quá mở, đã sửa quyền bằng `icacls`.
- Service trên EC2 ban đầu không tải được model vì thiếu AWS credentials, đã cấu hình `~/.aws/credentials` và `~/.aws/config`.
- Service systemd ban đầu trỏ sai đường dẫn Python trong venv, đã sửa sang `/home/ubuntu/venv/bin/python3`.

Sau khi xử lý, pipeline CI/CD, DVC remote, model serving và các endpoint FastAPI đều hoạt động thành công.
