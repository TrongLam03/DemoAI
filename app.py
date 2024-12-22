from flask import Flask, render_template, request
import cv2
import numpy as np
from PIL import Image
import io
import base64
import requests  # Thư viện để gửi yêu cầu tới API

app = Flask(__name__)

# Biến lưu ảnh đã qua xử lý để dùng lại
processed_image = None

# Hàm gọi Google Vision API
def google_vision_analysis(image):
    api_key = "AIzaSyAPvjxOI6TXYzG5otTSEK79dWrQJAU4ZQU"
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"

    # Chuyển ảnh thành base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Tạo payload cho API
    payload = {
        "requests": [
            {
                "image": {"content": image_base64},
                "features": [{"type": "LABEL_DETECTION", "maxResults": 10}]
            }
        ]
    }

    # Gửi yêu cầu POST
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        labels = response.json().get("responses", [{}])[0].get("labelAnnotations", [])
        return [label["description"] for label in labels]
    else:
        return ["Error: Could not connect to Vision API"]

# Hàm thực hiện GrabCut trên ảnh đầu vào
def grabcut_segmentation(image):
    # Chuyển đổi ảnh đầu vào thành mảng NumPy để dễ xử lý.
    image_np = np.array(image)
    # giá trị ban đầu là 0 (chưa phân loại).
    mask = np.zeros(image_np.shape[:2], np.uint8)

    # Định nghĩa hình chữ nhật ban đầu cho vùng chứa đối tượng cần phân đoạn.
    # Ở đây vùng này cách các biên của ảnh 50 pixel từ mọi phía.
    rect = (50, 50, image_np.shape[1] - 100, image_np.shape[0] - 100)
    # Tạo hai mô hình nền (background model) và tiền cảnh (foreground model).
    bgdModel = np.zeros((1, 65), np.float64)  # Mô hình nền
    fgdModel = np.zeros((1, 65), np.float64)  # Mô hình tiền cảnh

    # Áp dụng thuật toán GrabCut để phân đoạn ảnh:
    # - image_np: Ảnh gốc.
    # - rect: Hình chữ nhật xác định vùng quan tâm (ROI)..
    # - 5: Số lần lặp thuật toán.
    cv2.grabCut(image_np, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Áp dụng mask lên ảnh gốc để chỉ giữ lại các pixel thuộc tiền cảnh.
    # mask[:, :, np.newaxis]: Thêm một chiều để khớp với số chiều của ảnh (RGB).
    segmented_image = image_np * mask[:, :, np.newaxis]

    # Trả về ảnh đã phân đoạn.
    return segmented_image

# Hàm thực hiện Watershed
def watershed_segmentation(image):
    # Chuyển đổi ảnh đầu vào thành mảng NumPy để xử lý.
    image_np = np.array(image)
    
    # Chuyển đổi ảnh sang thang độ xám để giảm số kênh màu, làm đơn giản hóa việc phân đoạn.
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    # Làm mờ ảnh bằng bộ lọc Gaussian để giảm nhiễu, giúp quá trình phân đoạn hiệu quả hơn.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Áp dụng ngưỡng Otsu để tạo ảnh nhị phân.
    # THRESH_BINARY_INV: Đảo ngược vùng sáng/tối (foreground trở thành màu trắng).
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Tách vùng tiền cảnh (foreground) và nền (background):
    # Tạo kernel (ma trận 3x3) cho các phép biến đổi hình thái học.
    kernel = np.ones((3, 3), np.uint8)
    
    # Mở (morphological opening) để loại bỏ các điểm nhiễu nhỏ.
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Xác định vùng nền chắc chắn bằng phép giãn (dilate).
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Tính khoảng cách từ các pixel foreground tới biên nền bằng Distance Transform.
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    
    # Xác định vùng tiền cảnh chắc chắn bằng ngưỡng tỷ lệ (70% giá trị max).
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # Chuyển đổi vùng tiền cảnh chắc chắn thành kiểu dữ liệu nguyên để sử dụng sau này.
    sure_fg = np.uint8(sure_fg)
    
    # Xác định vùng không rõ (unknown region) bằng cách lấy hiệu của vùng nền và vùng tiền cảnh.
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Tạo marker (nhãn) cho từng vùng:
    # Sử dụng Connected Components để gán nhãn cho vùng foreground.
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Tăng giá trị của marker lên 1 để đảm bảo vùng nền chắc chắn có giá trị là 1.
    markers = markers + 1
    
    # Gán giá trị 0 cho vùng không rõ, để watershed nhận biết và phân chia.
    markers[unknown == 255] = 0
    
    # Áp dụng thuật toán Watershed lên ảnh:
    # Watershed sẽ cập nhật `markers` với các đường biên của vùng bằng giá trị -1.
    cv2.watershed(image_np, markers)
    
    # Tô màu từng vùng (segmentation):
    segmented = np.zeros_like(image_np)  # Khởi tạo ảnh trống với cùng kích thước như ảnh gốc.
    for marker in np.unique(markers):
        if marker == 1:  # Skip vùng nền (background).
            continue
        # Tạo màu ngẫu nhiên cho từng vùng dựa trên nhãn (marker).
        segmented[markers == marker] = np.random.randint(0, 255, size=(3,))
    
    # Trả về ảnh đã phân đoạn.
    return segmented
	
@app.route("/", methods=["GET", "POST"])
def index():
    global processed_image

    if request.method == "POST":  
        file = request.files.get("file")
        method = request.form.get("method")

        if file:
            image = Image.open(file.stream).convert("RGB")
            processed_image = image

        elif processed_image:
            image = processed_image

        if image and method:
            if method == "GrabCut":
                segmented_image = grabcut_segmentation(image)
                processed_image = Image.fromarray(segmented_image)
                buf = io.BytesIO()
                processed_image.save(buf, format="PNG")
                img_str = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
                return render_template("index.html", result=img_str, method=method)

            elif method == "Watershed":
                segmented_image = watershed_segmentation(image)
                processed_image = Image.fromarray(segmented_image)
                buf = io.BytesIO()
                processed_image.save(buf, format="PNG")
                img_str = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
                return render_template("index.html", result=img_str, method=method)

            elif method == "GoogleVision":
                labels = google_vision_analysis(image)
                return render_template("index.html", result=", ".join(labels), method=method)

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
