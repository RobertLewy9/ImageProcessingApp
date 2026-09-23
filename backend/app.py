
import os
import json
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import jwt
import datetime
from functools import wraps
import uuid
import sys
import io

# 导入你的PyTorch模型相关模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'KXZ_Model')) # 移除此行

from KXZ_Model.model import DualBranchFlowerModel  # 确保路径正确
import torch
import torchvision.transforms as transforms
from PIL import Image

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局缓冲区来捕获启动时的 stdout/stderr
captured_startup_output = io.StringIO()
original_stdout = sys.stdout
original_stderr = sys.stderr

sys.stdout = captured_startup_output
sys.stderr = captured_startup_output

# JWT配置
app.config['SECRET_KEY'] = 'your_secret_key_here'  # 替换为你的密钥

# 文件上传配置
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
HISTORY_FOLDER = os.path.join(os.path.dirname(__file__), 'history')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HISTORY_FOLDER, exist_ok=True)

# 辅助函数：保存和加载用户历史记录
def get_user_history_path(username):
    return os.path.join(HISTORY_FOLDER, f'upload_history_{username}.json')

def load_user_history(username):
    history_path = get_user_history_path(username)
    if os.path.exists(history_path):
        with open(history_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_user_history(username, history):
    history_path = get_user_history_path(username)
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

# --- Model Loading ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'KXZ_Model', 'model')
CONFIG_PATH = os.path.join(MODEL_DIR, 'config.json')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pth') # Corrected model path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = None
IMG_SIZE = None # Define IMG_SIZE globally
CLASS_NAMES = None

try:
    app.logger.info(f"Config JSON path: {CONFIG_PATH}, exists: {os.path.exists(CONFIG_PATH)}")
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
        image_size_list = config['image_size']
        IMG_SIZE = tuple(image_size_list) # Convert list to tuple for transforms.Resize
        class_mapping = config['class_mapping'] # Correctly read class_mapping
        CLASS_NAMES = {v: k for k, v in class_mapping.items()} # Invert for easy lookup
        num_classes = len(CLASS_NAMES)

    # 检查主模型路径是否存在，并加载其权重
    app.logger.info(f"主模型 PTH 路径: {MODEL_PATH}, 是否存在: {os.path.exists(MODEL_PATH)}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"主模型文件未找到: {MODEL_PATH}")

    # 实例化双分支模型，它会在内部加载 ConvNeXt 和 Swin 模型
    model = DualBranchFlowerModel(num_classes=num_classes)
    # 加载预训练的best_model.pth权重到整个双分支模型
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    app.logger.info("主模型和分支模型加载成功！")

except FileNotFoundError as e:
    app.logger.error(f"Error loading model or config: {e}")
    model = None
except KeyError as e:
    app.logger.error(f"Error reading config.json: Missing key {e}")
    model = None
except Exception as e:
    app.logger.error(f"An unexpected error occurred during model loading: {e}")
    model = None

# Preprocessing for recognition (ensure IMG_SIZE is defined)
if IMG_SIZE:
    # Ensure transforms.Resize gets a single integer or a tuple of two integers
    resize_arg = IMG_SIZE if isinstance(IMG_SIZE, tuple) else (IMG_SIZE, IMG_SIZE)
    recognition_transform = transforms.Compose([
        transforms.Resize(resize_arg),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
else:
    app.logger.warning("Warning: IMG_SIZE not defined, recognition_transform will not be set.")
    recognition_transform = None

# 图像预处理
# Removed redundant 'preprocess' variable.

# 图像处理函数
def process_image(image_data, operation):
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return None

    if operation == "grayscale":
        processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif operation == "canny":
        processed_img = cv2.Canny(img, 100, 200)
    elif operation == "sobel":
        # Implement Sobel edge detection
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        # Combine gradients to get the magnitude
        processed_img = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0)
    elif operation == "laplacian":
        # Implement Laplacian edge detection
        processed_img = cv2.Laplacian(img, cv2.CV_64F)
        processed_img = cv2.convertScaleAbs(processed_img)
    elif operation == "gaussian_blur":
        # 实现高斯模糊
        try:
            # ksize 是高斯核的大小，必须是正奇数元组，例如 (5, 5)
            # sigmaX 是高斯核在X方向的标准差，如果为0，则根据ksize计算
            processed_img = cv2.GaussianBlur(img, (5, 5), 0)
        except Exception as e:
            app.logger.error(f"Error applying Gaussian blur: {e}")
            return None
    elif operation == "sharpen":
        # 实现图像锐化
        # 定义锐化核
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        processed_img = cv2.filter2D(img, -1, kernel)
    else:
        return None

    _, buffer = cv2.imencode('.png', processed_img)
    return base64.b64encode(buffer).decode('utf-8')

# 装饰器：验证JWT token
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']

        if not token:
            return jsonify({'message': 'Token is missing!'}), 401

        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = data['username'] # 从token中提取用户名
        except Exception as e:
            app.logger.error(f"Token validation error: {e}")
            return jsonify({'message': 'Token is invalid!', 'error': str(e)}), 401

        return f(current_user, *args, **kwargs) # 将current_user传递给被装饰的函数

    return decorated

# 模拟用户数据库
users = {
    'testuser': {'password': 'testpass'},
    'admin': {'password': 'adminpass'}
} # In a real app, use a database and hash passwords

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'message': 'Missing username or password'}), 400

    if username in users:
        return jsonify({'message': 'User already exists'}), 409

    users[username] = {'password': password} # In real app: hash password
    return jsonify({'message': 'User registered successfully'}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    user = users.get(username)

    if not user or user['password'] != password: # In real app: compare hashed passwords
        return jsonify({'message': 'Invalid credentials'}), 401

    token = jwt.encode({
        'username': username,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=60)
    }, app.config['SECRET_KEY'], algorithm='HS256')

    return jsonify({'token': token})

@app.route('/api/process-image', methods=['POST'])
@token_required
def handle_image(current_user):
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    operation = request.form.get('operation', 'grayscale')

    if not file.filename:
        return jsonify({'error': 'No selected image file'}), 400

    try:
        image_data = file.read()
        processed_image_b64 = process_image(image_data, operation)
        
        if processed_image_b64 is None:
            return jsonify({'error': f'Failed to process image with operation: {operation}'}), 500

        # 保存原始图片
        original_filename = f'{uuid.uuid4().hex}_{file.filename}'
        original_image_path = os.path.join(UPLOAD_FOLDER, original_filename)
        Image.open(io.BytesIO(image_data)).save(original_image_path)

        # 保存处理后的图片
        processed_filename = f'{uuid.uuid4().hex}_{operation}.png'
        processed_image_full_path = os.path.join(UPLOAD_FOLDER, processed_filename)
        decoded_processed_img = base64.b64decode(processed_image_b64)
        with open(processed_image_full_path, 'wb') as f:
            f.write(decoded_processed_img)

        # 记录到用户历史
        history = load_user_history(current_user)
        history_entry = {
            'id': str(uuid.uuid4()),
            'original_filename': file.filename,
            'upload_time': datetime.datetime.now().isoformat(),
            'operation_type': operation,
            'original_image_path': original_image_path,
            'processed_image_path': processed_image_full_path,
            'recognition_result': None, # 图像处理没有识别结果
            'heatmap_image_path': None # 图像处理没有热力图
        }
        history.insert(0, history_entry) # 最新记录在前面
        save_user_history(current_user, history)

        return jsonify({'processed_image': processed_image_b64, 'history_id': history_entry['id']})

    except Exception as e:
        app.logger.error(f"Error processing image: {e}")

        return jsonify({'error': f'Image processing failed: {str(e)}'}), 500

@app.route('/api/recognize-image', methods=['POST'])
@token_required
def recognize_image(current_user):
    if model is None or recognition_transform is None or CLASS_NAMES is None:
        app.logger.error("Model not loaded or not properly configured for recognition.")
        return jsonify({'error': 'Model not loaded or not properly configured.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if not file.filename:
        return jsonify({'error': 'No selected image file'}), 400

    try:
        image_data = file.read()
        img_pil = Image.open(io.BytesIO(image_data)).convert('RGB')

        # 保存原始图片
        original_filename = f'{uuid.uuid4().hex}_{file.filename}'
        original_image_path = os.path.join(UPLOAD_FOLDER, original_filename)
        Image.open(io.BytesIO(image_data)).save(original_image_path)

        # 使用 preprocess_image_with_padding（如果存在）
        # if hasattr(sys.modules['utils'], 'preprocess_image_with_padding'):
        #     preprocessed_img_pil = sys.modules['utils'].preprocess_image_with_padding(img_pil, IMG_SIZE) # 修正调用方式
        # else:
        preprocessed_img_pil = img_pil

        # 应用标准变换，并确保可计算梯度
        input_tensor = recognition_transform(preprocessed_img_pil).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True) # 确保输入张量可以计算梯度

        # 第一次前向传播，获取预测结果
        with torch.no_grad():
            logits, _ = model(input_tensor) # DualBranchFlowerModel 返回 logits 和 fused_features
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
            top5_prob, top5_indices = torch.topk(probabilities, 5)
            
        # 获取预测结果（不变）
        predictions = []
        for i in range(top5_prob.size(0)):
            class_idx = top5_indices[i].item()
            class_name = CLASS_NAMES.get(class_idx, f'Unknown Class {class_idx}')
            predictions.append({'class': class_name, 'probability': top5_prob[i].item()})
        
        # --- 热力图生成逻辑开始 ---
        # 清零之前的梯度
        # model.zero_grad()
        
        # 第二次前向传播，并获取目标类别的梯度
        # logits_for_grad, _ = model(input_tensor) # 再次前向传播以获取梯度
        # target_logit = logits_for_grad[0, top5_indices[0]] # 获取最高预测类别的logit
        
        # target_logit.backward() # 反向传播以获取梯度
        
        # 获取输入图像的梯度
        # gradients = input_tensor.grad.abs().mean(dim=1, keepdim=True) # 取绝对值并求平均，得到单通道梯度图
        
        # 归一化梯度图到 0-1 范围
        # gradients = gradients.squeeze().cpu().numpy() # 移除批次和通道维度，转为numpy
        # gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)
        
        # 上采样到原始图像大小 (IMG_SIZE 是一个元组 (height, width))
        # heatmap = cv2.resize(gradients, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        
        # 将热力图颜色化
        # heatmap = np.uint8(255 * heatmap)
        # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # 使用 JET 色谱
        
        # 保存热力图
        # heatmap_filename = f'{uuid.uuid4().hex}_heatmap.png'
        # heatmap_image_path = os.path.join(UPLOAD_FOLDER, heatmap_filename)
        # cv2.imwrite(heatmap_image_path, heatmap)
        
        # 将热力图转换为 base64 编码
        # _, buffer = cv2.imencode('.png', heatmap)
        # heatmap_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # --- 热力图生成逻辑结束 ---

        # 记录到用户历史
        history = load_user_history(current_user)
        history_entry = {
            'id': str(uuid.uuid4()),
            'original_filename': file.filename,
            'upload_time': datetime.datetime.now().isoformat(),
            'operation_type': 'recognize',
            'original_image_path': original_image_path,
            'processed_image_path': None, # 图像识别没有处理图片
            'recognition_result': predictions,
            'heatmap_image_path': None # 图像识别没有热力图
        }
        history.insert(0, history_entry) # 最新记录在前面
        save_user_history(current_user, history)
        
        # 保存识别结果为 JSON 文件
        recognition_filename = f'{uuid.uuid4().hex}_recognition.json'
        recognition_file_path = os.path.join(UPLOAD_FOLDER, recognition_filename)
        with open(recognition_file_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=4)
        history_entry['recognition_file_path'] = recognition_file_path # 更新历史记录条目
        save_user_history(current_user, history)

        return jsonify({'predictions': predictions, 'history_id': history_entry['id']}) # 移除heatmap_image

    except Exception as e:
        app.logger.error(f"Error recognizing image: {e}")

        return jsonify({'error': f'Image recognition failed: {str(e)}'}), 500

@app.route('/api/history', methods=['GET'])
@token_required
def get_history(current_user):
    try:
        history = load_user_history(current_user)
        # 为了安全和效率，不直接返回完整的本地文件路径，而是返回可以通过API访问的相对路径
        # 前端需要通过另一个API来获取这些图片
        sanitized_history = []
        for entry in history:
            new_entry = entry.copy()
            # 将本地路径转换为前端可用的URL
            if new_entry.get('original_image_path'):
                new_entry['original_image_url'] = f'/api/uploads/{os.path.basename(new_entry["original_image_path"])}'
                del new_entry['original_image_path']
            if new_entry.get('processed_image_path'):
                new_entry['processed_image_url'] = f'/api/uploads/{os.path.basename(new_entry["processed_image_path"])}'
                del new_entry['processed_image_path']
            if new_entry.get('heatmap_image_path'):
                new_entry['heatmap_image_url'] = f'/api/uploads/{os.path.basename(new_entry["heatmap_image_path"])}'
                del new_entry['heatmap_image_path']
            if new_entry.get('recognition_file_path'): # 新增：处理识别结果文件路径
                new_entry['recognition_file_url'] = f'/api/uploads/{os.path.basename(new_entry["recognition_file_path"])}'
                del new_entry['recognition_file_path']
            sanitized_history.append(new_entry)
        return jsonify(sanitized_history)
    except Exception as e:
        app.logger.error(f"Error getting history for user {current_user}: {e}")

        return jsonify({'error': f'Failed to retrieve history: {str(e)}'}), 500

@app.route('/api/history/<string:history_id>', methods=['DELETE'])
@token_required
def delete_history_entry(current_user, history_id):
    try:
        history = load_user_history(current_user)
        entry_to_delete = None
        entry_index = -1

        for i, entry in enumerate(history):
            if entry['id'] == history_id:
                entry_to_delete = entry
                entry_index = i
                break

        if entry_to_delete is None:
            # 幂等性：即使找不到也返回200，便于前端多次操作不报错
            return jsonify({'message': 'Already deleted or not found.'}), 200

        # 删除物理文件
        for key in ['original_image_path', 'processed_image_path', 'heatmap_image_path', 'recognition_file_path']:
            file_path = entry_to_delete.get(key)
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                app.logger.info(f"Deleted file: {file_path}")

        # 从历史记录中移除条目并保存
        del history[entry_index]
        save_user_history(current_user, history)

        return jsonify({'message': 'History entry deleted successfully.'}), 200
    except Exception as e:
        app.logger.error(f"Error deleting history entry {history_id} for user {current_user}: {e}")
        return jsonify({'error': f'Failed to delete history entry: {str(e)}'}), 500

@app.route('/api/history/all', methods=['DELETE'])
@token_required
def delete_all_history(current_user):
    try:
        history = load_user_history(current_user)
        # 删除所有物理文件
        for entry in history:
            for key in ['original_image_path', 'processed_image_path', 'heatmap_image_path', 'recognition_file_path']:
                file_path = entry.get(key)
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
        # 清空历史
        save_user_history(current_user, [])
        return jsonify({'message': 'All history cleared successfully.'}), 200
    except Exception as e:
        app.logger.error(f"Error clearing all history for user {current_user}: {e}")
        return jsonify({'error': f'Failed to clear history: {str(e)}'}), 500

@app.route('/api/change-password', methods=['POST'])
@token_required
def change_password(current_user):
    data = request.json
    old_password = data.get('old_password')
    new_password = data.get('new_password')

    if not old_password or not new_password:
        return jsonify({'message': 'Missing old password or new password'}), 400

    user = users.get(current_user)
    if not user or user['password'] != old_password: # 在实际应用中应比较哈希密码
        return jsonify({'message': 'Invalid old password'}), 401

    users[current_user]['password'] = new_password # 更新密码
    return jsonify({'message': 'Password changed successfully'}), 200

# Serve static files from the UPLOAD_FOLDER
@app.route('/api/uploads/<filename>')
@token_required
def uploaded_file(current_user, filename):
    # 这里可以添加验证，确保用户只能访问自己的文件
    return send_from_directory(UPLOAD_FOLDER, filename)

# Serve static files from the frontend build directory
@app.route('/')
def serve_index():
    return send_from_directory('../frontend/dist', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('../frontend/dist', path)

if __name__ == '__main__':
    # 在 app.run() 之前恢复 stdout 和 stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    # 将启动时捕获的输出记录到 Flask 的 logger 中
    startup_logs = captured_startup_output.getvalue().strip()
    if startup_logs:
        app.logger.info("--- Captured Startup Logs ---")
        for line in startup_logs.splitlines():
            app.logger.info(line)
        app.logger.info("---------------------------")

    app.run(debug=False)
