from flask import Flask, render_template, Response, request, redirect, url_for
import cv2

app = Flask(__name__)

# Khởi tạo biến toàn cục
tracker = None
video_capture = None
tracking_bbox = None
object_type = None  # Biến để lưu loại đối tượng (Người hoặc Vật)

# Hàm khởi tạo bộ theo dõi
def initialize_tracker(frame, bbox):
    global tracker
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)

# Hàm để tạo luồng video
def generate_frames():
    global video_capture, tracker, tracking_bbox, object_type
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Nếu bộ theo dõi đã được khởi tạo, cập nhật và vẽ lên đối tượng
        if tracker is not None:
            ok, tracking_bbox = tracker.update(frame)
            if ok:
                (x, y, w, h) = [int(v) for v in tracking_bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
                # Hiển thị chú thích loại đối tượng
                cv2.putText(frame, object_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            else:
                cv2.putText(frame, "Không thể theo dõi đối tượng", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Mã hóa khung hình thành JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Trả về luồng video dạng multipart
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route chính
@app.route('/', methods=['GET', 'POST'])
def index():
    global video_capture, tracking_bbox, object_type

    if request.method == 'POST':
        file = request.files['file']
        if file:
            video_path = './uploaded_video.mp4'
            file.save(video_path)  # Lưu video tạm thời

            # Mở video
            video_capture = cv2.VideoCapture(video_path)
            
            # Lấy khung hình đầu tiên và chọn đối tượng
            ret, frame = video_capture.read()
            if ret:
                # Chọn vùng đối tượng
                bbox = cv2.selectROI("Chọn đối tượng", frame, fromCenter=False, showCrosshair=True)
                cv2.destroyAllWindows()
                
                if bbox != (0, 0, 0, 0):  # Kiểm tra nếu người dùng đã chọn vùng
                    tracking_bbox = bbox
                    initialize_tracker(frame, bbox)

                    # Yêu cầu người dùng nhập loại đối tượng (Người hoặc Vật)
                    object_type = request.form.get('object_type', 'Vật')

            return redirect(url_for('video_feed'))  # Chuyển đến luồng video

    return render_template('vd.html')

# Route để phát video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
