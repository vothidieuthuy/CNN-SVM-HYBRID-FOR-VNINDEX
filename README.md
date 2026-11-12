# Dự báo VN-Index bằng CNN–SVM

Mã nguồn thực nghiệm cho bài báo COMBELT 2025 (*CNN-SVM*). Hai notebook dự báo **giá đóng cửa VN-Index của phiên kế tiếp** và so sánh các mô hình ARIMA/ARIMAX, SVR, CNN-1D, CNN-2D và hai mô hình lai CNN + SVR.

| Notebook | Đặc trưng đầu vào cho mô hình học máy | Biến ngoại sinh của ARIMAX |
|---|---|---|
| [ohclv-only_code.ipynb](ohclv-only_code.ipynb) | Chỉ 5 cột OHLCV: `Open, High, Low, Close, Volume` | `Open, High, Low, Volume` |
| [ohclv-ta_code.ipynb](ohclv-ta_code.ipynb) | OHLCV cùng toàn bộ chỉ báo kỹ thuật do `ta.add_all_ta_features` sinh ra (khoảng 90 cột) | `Open, High, Low, Volume, trend_macd, momentum_rsi, volume_sma_em` |

Ngoài khác biệt về đặc trưng, hai notebook dùng chung quy trình. Số epoch tiền huấn luyện CNN cũng khác nhau (xem mục 3).

---

## 1. Dữ liệu

Notebook đọc file **`VNI.csv`** đặt cùng thư mục. File này **không có sẵn trong thư mục**, cần bổ sung trước khi chạy.

Cấu trúc file mà code yêu cầu:

- Các cột bị bỏ: `Unnamed: 0.1`, `Unnamed: 0`, `Market`, `Symbol`, `Time`, `Value`.
- Cột `Date` ở định dạng `DD/MM/YYYY`.
- Sau khi bỏ các cột trên, 5 cột còn lại phải xếp **đúng thứ tự** Open, High, Low, Close, Volume, vì code đổi tên theo vị trí (`data.columns = [...]`).

Phạm vi dữ liệu dùng: **16/07/2019 – 13/09/2024**, tổng cộng 1.293 phiên. Dữ liệu được chia theo thời gian, **80% đầu để huấn luyện và 20% cuối để kiểm tra**.

## 2. Môi trường

Python 3 cùng các thư viện:

```
pandas numpy matplotlib seaborn scipy
scikit-learn statsmodels
torch
ta              # thư viện chỉ báo kỹ thuật (Technical Analysis)
torchsummary    # in kiến trúc mạng
torchviz        # chỉ ohclv-ta_code import (không dùng thực tế)
```

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels torch ta torchsummary torchviz
```

Seed được cố định bằng `set_seed(42)` (Python, NumPy, PyTorch, cuDNN deterministic) để kết quả lặp lại được.

## 3. Quy trình trong notebook

| Bước | Nội dung |
|---|---|
| 1. Nạp dữ liệu | Đọc `VNI.csv`, tính chỉ báo kỹ thuật bằng `add_all_ta_features(fillna=True)`, rồi vẽ chuỗi giá Close với ranh giới train/test |
| 2. Thống kê mô tả | Thống kê mô tả, độ lệch (skewness), độ nhọn (kurtosis) và histogram của Close |
| 3. Kiểm định | ADF (tính dừng), Shapiro–Wilk và Jarque–Bera (phân phối chuẩn); ACF/PACF của chuỗi gốc và chuỗi sai phân bậc 1 |
| 4. ARIMA / ARIMAX | `ARIMA(1,1,1)` và `SARIMAX(1,1,1)` có biến ngoại sinh, dự báo nhiều bước cho toàn bộ tập test; vẽ 20 bước đầu |
| 5. Tiền xử lý | `MinMaxScaler` cho X và y; nhãn là Close của phiên t+1; cửa sổ trượt `window_size = 10` phiên |
| 6. Mô hình CNN | **CNN-1D**: `Conv1d(64, k=2)` → ReLU → `MaxPool1d(2)` → Flatten → Linear. **CNN-2D**: `Conv2d(64, 3×3, padding=1)` → ReLU → `MaxPool2d(2×2)` → Flatten → Linear |
| 7. Huấn luyện CNN | Adam (lr = 0.001), MSE loss, full-batch. Số epoch: **20** ở `ohclv-only`, **200** ở `ohclv-ta` |
| 8. CNN + SVR | Lấy vector sau lớp Flatten của CNN đã huấn luyện làm đặc trưng cho `SVR(kernel='rbf')` |
| 9. SVR gốc | `SVR(kernel='rbf')` trên cửa sổ đặc trưng đã duỗi phẳng |
| 10. Đánh giá | Đưa kết quả về thang giá gốc, tính **RMSE** và **MAE**, rồi vẽ đồ thị so sánh mọi mô hình trên tập test |

## 4. Kết quả (đầu ra đã lưu trong notebook)

| Mô hình | OHLCV RMSE | OHLCV MAE | OHLCV + TA RMSE | OHLCV + TA MAE |
|---|---:|---:|---:|---:|
| ARIMA | 68.62 | 60.63 | 68.62 | 60.63 |
| ARIMAX | 65.96 | 58.14 | 65.55 | 58.22 |
| SVR | 46.10 | 40.59 | 30.02 | 22.19 |
| CNN-1D + SVR | **32.22** | **26.57** | 26.78 | **21.15** |
| CNN-2D + SVR | 36.85 | 32.41 | **26.15** | 21.29 |
| CNN-1D | 126.14 | 120.33 | 28.33 | 23.04 |
| CNN-2D | 56.26 | 46.72 | 27.56 | 22.67 |

*OHLCV huấn luyện CNN 20 epoch; OHLCV + TA huấn luyện 200 epoch.*

Ô markdown cuối mỗi notebook ghi thêm kết quả khi thay số epoch: 20, 50, 100, 500, 1000, 2000 và 5000. Tiêu đề của ô này là "Out_channel = 16", trong khi code hiện tại dùng 64 kênh.

## 5. Lưu ý khi chạy lại

- **Chạy tuần tự từ trên xuống, nhưng có hai ngoại lệ:**
  - `ohclv-only_code.ipynb`: **thiếu đoạn huấn luyện SVR gốc** (`svm_raw`, mục 7.3), trong khi bước 8 dùng biến `y_pred_svm`. Kết quả SVR đã lưu trong notebook có được nhờ trạng thái kernel từ lần chạy trước. Muốn chạy lại từ đầu, chép đoạn 7.3 từ `ohclv-ta_code.ipynb` vào sau mục 7.2.
  - `ohclv-ta_code.ipynb`: ô in kiến trúc (`torchsummary`, ô thứ 8) đứng **trước** ô định nghĩa mô hình và còn gọi tới các lớp `CNN1D`, `CNN2D` chưa được định nghĩa. Hãy bỏ qua ô này, hoặc chạy nó sau ô định nghĩa mô hình và chỉ giữ hai dòng `CNN1D_Only` và `CNN2D_Only`.
- Ô 8 trong `ohclv-only_code.ipynb` đã bị comment toàn bộ (phiên bản cũ của bước căn chỉnh). Bước này hiện nằm trong ô huấn luyện.
- Hai đặc điểm của cách chia dữ liệu nên biết khi diễn giải kết quả:
  - `MinMaxScaler` được fit trên **toàn bộ** chuỗi, gồm cả tập test.
  - Code nối thêm 2 mẫu train (`X_train[-3:-1]`) vào cuối tập test, rồi cắt mọi dự báo theo phần đuôi (`min_len`) để các mảng có cùng độ dài.
- ARIMA/ARIMAX dự báo **nhiều bước liên tiếp** cho cả giai đoạn test. Các mô hình học máy dự báo **một bước** dựa trên cửa sổ 10 phiên có dữ liệu thực.
