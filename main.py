from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf

# Tải mô hình khi khởi động
model = tf.keras.models.load_model("flower_model.h5")

# Nhãn tương ứng với các lớp
flower_labels = [
    'hoa_anh_dao', 'hoa_cam_tu_cau', 'hoa_cuc', 'hoa_hong',
    'hoa_huong_duong', 'hoa_lan', 'hoa_ly',
    'hoa_oai_huong', 'hoa_sen', 'hoa_tulip'
]

# Mapping từ tên kỹ thuật sang tên hiển thị dễ đọc
flower_display_names = {
    'hoa_anh_dao': 'Hoa anh đào',
    'hoa_cam_tu_cau': 'Hoa cẩm tú cầu',
    'hoa_cuc': 'Hoa cúc',
    'hoa_hong': 'Hoa hồng',
    'hoa_huong_duong': 'Hoa hướng dương',
    'hoa_lan': 'Hoa lan',
    'hoa_ly': 'Hoa ly',
    'hoa_oai_huong': 'Hoa oải hương',
    'hoa_sen': 'Hoa sen',
    'hoa_tulip': 'Hoa tulip'
}

# Khởi tạo ứng dụng FastAPI
app = FastAPI(title="API Nhận Diện Loài Hoa",
              description="API để nhận diện loài hoa từ hình ảnh")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả nguồn truy cập
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hàm xử lý ảnh và dự đoán loài hoa
def predict_flower(image: Image.Image):
    # Resize ảnh về đúng kích thước mà mô hình yêu cầu
    image = image.resize((224, 224))

    # Chuyển sang mảng numpy và chuẩn hóa
    image_array = np.array(image) / 255.0

    # Nếu là ảnh xám thì chuyển thành 3 kênh RGB
    if image_array.ndim == 2:
        image_array = np.stack((image_array,) * 3, axis=-1)

    # Nếu có nhiều kênh hơn 3 thì giữ lại 3 kênh đầu
    if image_array.shape[-1] != 3:
        image_array = image_array[..., :3]

    # Thêm chiều batch
    image_array = np.expand_dims(image_array, axis=0)

    # Dự đoán
    prediction = model.predict(image_array)
    predicted_index = np.argmax(prediction)

    # Lấy nhãn kỹ thuật và tên hiển thị
    flower_label = flower_labels[predicted_index]
    flower_display = flower_display_names.get(flower_label, flower_label)

    return flower_label, flower_display

@app.get("/")
def read_root():
    """Endpoint mặc định hiển thị thông tin API"""
    return {
        "message": "API Nhận Diện Loài Hoa",
        "usage": "Gửi POST request với hình ảnh đến /predict"
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Endpoint để dự đoán loài hoa từ hình ảnh tải lên"""
    if not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={"error": "File tải lên phải là hình ảnh"}
        )

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        flower_label, flower_display = predict_flower(image)

        return {
            "label": flower_label,
            "name": flower_display
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Đã xảy ra lỗi: {str(e)}"}
        )

# Chạy server nếu chạy trực tiếp file này
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
