from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import cv2
import numpy as np
import os
import base64
from datetime import datetime, timedelta
import face_recognition
import json
from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof
import time
from src.resource_monitor import ResourceMonitor

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Initialize face detection models
face_detector = YOLOv5('saved_models/yolov5s-face.onnx')
anti_spoof = AntiSpoof('saved_models/AntiSpoofing_bin_1.5_128.onnx')

# Create necessary directories
if not os.path.exists('new_images'):
    os.makedirs('new_images')
if not os.path.exists('database'):
    os.makedirs('database')

# Path to store user data
USER_DB_PATH = 'database/users.json'

# Path to store exam data
EXAM_DB_PATH = 'database/exam_data.json'

# Path to store monitoring data
MONITORING_DB_PATH = 'database/monitoring_data.json'

def load_users():
    try:
        if os.path.exists(USER_DB_PATH):
            with open(USER_DB_PATH, 'r', encoding='utf-8') as f:
                users = json.load(f)
                print(f"Loaded users: {users}")  # Debug log
                return users
        print("User database file not found")  # Debug log
        return {}
    except Exception as e:
        print(f"Error loading users: {str(e)}")  # Debug log
        return {}

def save_users(users):
    with open(USER_DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=4)

def load_exam_data():
    if os.path.exists(EXAM_DB_PATH):
        with open(EXAM_DB_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'groups': {}}

def save_exam_data(data):
    with open(EXAM_DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_monitoring_data():
    """Load monitoring data from file with proper error handling."""
    try:
        if os.path.exists(MONITORING_DB_PATH):
            with open(MONITORING_DB_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Ensure required structure exists
                if not isinstance(data, dict):
                    data = {}
                if 'activities' not in data:
                    data['activities'] = {}
                if 'resources' not in data:
                    data['resources'] = {}
                if 'notifications' not in data:
                    data['notifications'] = {}
                return data
        # If file doesn't exist, create default structure
        default_data = {
            'activities': {},
            'resources': {},
            'notifications': {}
        }
        save_monitoring_data(default_data)
        return default_data
    except Exception as e:
        print(f"Error loading monitoring data: {str(e)}")
        # Return default structure on error
        return {
            'activities': {},
            'resources': {},
            'notifications': {}
        }

def save_monitoring_data(data):
    """Save monitoring data with proper error handling."""
    try:
        # Ensure database directory exists
        os.makedirs(os.path.dirname(MONITORING_DB_PATH), exist_ok=True)
        
        # Validate data structure
        if not isinstance(data, dict):
            raise ValueError("Invalid data structure")
            
        # Ensure required keys exist
        data.setdefault('activities', {})
        data.setdefault('resources', {})
        data.setdefault('notifications', {})
        
        # Save with pretty formatting for readability
        with open(MONITORING_DB_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
    except Exception as e:
        print(f"Error saving monitoring data: {str(e)}")
        raise

def increased_crop(img, bbox, bbox_inc=1.5):
    real_h, real_w = img.shape[:2]
    x, y, w, h = bbox
    w, h = w - x, h - y
    l = max(w, h)
    xc, yc = x + w/2, y + h/2
    x, y = int(xc - l*bbox_inc/2), int(yc - l*bbox_inc/2)
    x1 = 0 if x < 0 else x 
    y1 = 0 if y < 0 else y
    x2 = real_w if x + l*bbox_inc > real_w else x + int(l*bbox_inc)
    y2 = real_h if y + l*bbox_inc > real_h else y + int(l*bbox_inc)
    img = img[y1:y2,x1:x2,:]
    img = cv2.copyMakeBorder(img, 
                            y1-y, int(l*bbox_inc-y2+y), 
                            x1-x, int(l*bbox_inc)-x2+x, 
                            cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

def verify_real_face(image_data):
    try:
        # Convert base64 to image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        img_data = base64.b64decode(image_data)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return False, None, "ไม่สามารถแปลงรูปภาพได้"
            
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect face
        pred = face_detector([img_rgb])[0]
        if pred.shape[0] == 0:
            return False, None, "ไม่พบใบหน้าในภาพ กรุณาหันหน้าเข้ากล้อง"
            
        # Get face bbox
        bbox = pred.flatten()[:4].astype(int)
        
        # Ensure face is centered and large enough
        img_height, img_width = img.shape[:2]
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        face_center_x = (bbox[0] + bbox[2]) / 2
        face_center_y = (bbox[1] + bbox[3]) / 2
        
        # Check if face is too small
        min_face_size = min(img_width, img_height) * 0.2  # Face should be at least 20% of image
        if face_width < min_face_size or face_height < min_face_size:
            return False, None, "กรุณาเข้าใกล้กล้องมากขึ้น"
            
        # Check if face is centered
        center_threshold = 0.2  # Face center should be within 20% of image center
        if abs(face_center_x - img_width/2) > img_width * center_threshold or \
           abs(face_center_y - img_height/2) > img_height * center_threshold:
            return False, None, "กรุณาจัดให้ใบหน้าอยู่กลางภาพ"
        
        # Anti-spoofing check with increased crop
        face_img = increased_crop(img_rgb, bbox, bbox_inc=1.5)
        pred = anti_spoof([face_img])[0]
        score = pred[0][0]
        label = np.argmax(pred)
        
        if label == 0 and score > 0.5:
            return True, img, bbox
        else:
            return False, None, "กรุณาใช้ใบหน้าจริงในการยืนยันตัวตน"
            
    except Exception as e:
        print(f"Error in verify_real_face: {str(e)}")  # Add debug log
        return False, None, f"เกิดข้อผิดพลาด: {str(e)}"

@app.route('/verify_face', methods=['POST'])
def verify_face():
    try:
        image_data = request.json.get('image_data', '')
        if not image_data:
            return jsonify({'success': False, 'message': 'ไม่พบข้อมูลรูปภาพ'})
            
        success, img, result = verify_real_face(image_data)
        if not success:
            return jsonify({'success': False, 'message': result})
            
        # If face is real, check for matching face in database
        face_locations = face_recognition.face_locations(img)
        if not face_locations:
            return jsonify({'success': False, 'message': 'ไม่พบใบหน้าในภาพ'})
            
        login_face_encoding = face_recognition.face_encodings(img, face_locations)[0]
        
        # Load users and compare faces
        users = load_users()
        if not users:
            return jsonify({'success': False, 'message': 'ไม่พบข้อมูลผู้ใช้ในระบบ'})
            
        for user_id, user_data in users.items():
            stored_image_path = os.path.join('new_images', user_data['face_image'])
            if not os.path.exists(stored_image_path):
                continue
                
            try:
                stored_img = face_recognition.load_image_file(stored_image_path)
                stored_face_locations = face_recognition.face_locations(stored_img)
                if not stored_face_locations:
                    continue
                    
                stored_face_encoding = face_recognition.face_encodings(stored_img, stored_face_locations)[0]
                
                # Compare faces with lower tolerance for higher accuracy
                if face_recognition.compare_faces([stored_face_encoding], login_face_encoding, tolerance=0.45)[0]:
                    # Return user data for form submission
                    return jsonify({
                        'success': True,
                        'message': 'ยืนยันตัวตนสำเร็จ',
                        'user_id': user_id,
                        'name': user_data['name']
                    })
                    
            except Exception as e:
                print(f"Error comparing faces for user {user_id}: {str(e)}")
                continue
                
        return jsonify({'success': False, 'message': 'ไม่พบข้อมูลใบหน้าที่ตรงกัน กรุณาลองใหม่อีกครั้ง'})
        
    except Exception as e:
        print(f"Error in verify_face route: {str(e)}")
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_id = request.form.get('user_id', '').strip()
        name = request.form.get('name', '').strip()
        position = request.form.get('position', '').strip()
        image_data = request.form.get('image_data', '')
        
        if not user_id or not name or not position:
            flash('กรุณากรอกข้อมูลให้ครบถ้วน')
            return redirect(url_for('register'))
        
        users = load_users()
        if user_id in users:
            flash('รหัสผู้ใช้นี้มีอยู่แล้ว!')
            return redirect(url_for('register'))
        
        if not image_data:
            flash('กรุณาถ่ายภาพใบหน้า!')
            return redirect(url_for('register'))
            
        # Save face image
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            img_data = base64.b64decode(image_data)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                flash('ไม่สามารถแปลงรูปภาพได้')
                return redirect(url_for('register'))
                
            image_filename = f'face_{user_id}.jpg'
            image_path = os.path.join('new_images', image_filename)
            cv2.imwrite(image_path, img)
            
            # Save user data with position
            users[user_id] = {
                'name': name,
                'position': position,
                'face_image': image_filename
            }
            save_users(users)
            
            flash('ลงทะเบียนสำเร็จ! กรุณาเข้าสู่ระบบ')
            return redirect(url_for('login'))
            
        except Exception as e:
            flash(f'เกิดข้อผิดพลาด: {str(e)}')
            return redirect(url_for('register'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            user_id = request.form.get('user_id')
            name = request.form.get('name')
            
            print(f"Login attempt - User ID: {user_id}, Name: {name}")  # Debug log
            
            if not user_id or not name:
                print("Missing user_id or name in form data")  # Debug log
                flash('เกิดข้อผิดพลาดในการยืนยันตัวตน กรุณาลองใหม่')
                return redirect(url_for('login'))

            # Verify user exists in database
            users = load_users()
            print(f"Loaded users from database: {list(users.keys())}")  # Debug log
            
            if user_id not in users:
                print(f"User ID {user_id} not found in database")  # Debug log
                flash('ข้อมูลผู้ใช้ไม่ถูกต้อง กรุณาลองใหม่')
                return redirect(url_for('login'))
                
            if users[user_id]['name'] != name:
                print(f"Name mismatch for user {user_id}")  # Debug log
                flash('ข้อมูลผู้ใช้ไม่ถูกต้อง กรุณาลองใหม่')
                return redirect(url_for('login'))
                
            # Set session data including position
            session['user_id'] = user_id
            session['name'] = name
            session['position'] = users[user_id]['position']
            print(f"Successfully set session for user {user_id}")  # Debug log
            
            return redirect(url_for('main'))
            
        except Exception as e:
            print(f"Error in login route: {str(e)}")  # Debug log
            flash(f'เกิดข้อผิดพลาดในระบบ: {str(e)}')
            return redirect(url_for('login'))
        
    return render_template('login.html')

@app.route('/main')
def main():
    if 'user_id' not in session:
        flash('กรุณาเข้าสู่ระบบก่อน')
        return redirect(url_for('login'))
    
    try:
        # Load user data
        users = load_users()
        user_id = session['user_id']
        
        if user_id not in users:
            session.clear()
            flash('ไม่พบข้อมูลผู้ใช้')
            return redirect(url_for('login'))
            
        return render_template('main.html', 
                             user_id=session['user_id'],
                             name=session['name'],
                             position=session['position'])
    except Exception as e:
        print(f"Error in main route: {str(e)}")  # Debug log
        session.clear()
        flash('เกิดข้อผิดพลาดในระบบ กรุณาเข้าสู่ระบบใหม่')
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.clear()
    flash('คุณได้ออกจากระบบแล้ว')
    return redirect(url_for('index'))

@app.route('/api/get_teachers', methods=['GET'])
def get_teachers():
    try:
        users = load_users()
        print(f"Users data in get_teachers: {users}")  # Debug log
        
        if not users:
            print("No users found in database")  # Debug log
            return jsonify({
                'success': False,
                'message': 'ไม่พบข้อมูลผู้ใช้ในระบบ'
            })

        teachers = []
        for user_id, user_data in users.items():
            print(f"Checking user {user_id}: {user_data}")  # Debug log
            if user_data.get('position') == 'อาจารย์':
                teachers.append({
                    'id': user_id,
                    'name': user_data['name']
                })

        print(f"Found teachers: {teachers}")  # Debug log

        if not teachers:
            print("No teachers found")  # Debug log
            return jsonify({
                'success': False,
                'message': 'ไม่พบรายชื่ออาจารย์ในระบบ'
            })

        return jsonify({
            'success': True,
            'teachers': teachers
        })
    except Exception as e:
        print(f"Error in get_teachers: {str(e)}")  # Debug log
        return jsonify({
            'success': False,
            'message': f'ไม่สามารถโหลดรายชื่ออาจารย์ได้: {str(e)}',
            'error': str(e)
        })

@app.route('/api/get_teacher_groups/<teacher_id>')
def get_teacher_groups(teacher_id):
    try:
        exam_data = load_exam_data()
        teacher_groups = exam_data['groups'].get(teacher_id, {})
        
        # Format groups data for frontend
        groups = []
        for group_id, group_data in teacher_groups.items():
            groups.append({
                'id': group_id,
                'name': group_data['name']
            })
            
        return jsonify({
            'success': True,
            'groups': groups
        })
    except Exception as e:
        print(f"Error in get_teacher_groups: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'ไม่สามารถโหลดข้อมูลกลุ่มได้',
            'error': str(e)
        })

@app.route('/api/groups/<teacher_id>')
def get_teacher_tables(teacher_id):
    try:
        exam_data = load_exam_data()
        if teacher_id not in exam_data['groups']:
            return jsonify({
                'success': False,
                'message': 'ไม่พบข้อมูลกลุ่มของอาจารย์'
            })
            
        return jsonify(exam_data['groups'][teacher_id])
    except Exception as e:
        print(f"Error in get_teacher_tables: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'ไม่สามารถโหลดข้อมูลโต๊ะได้',
            'error': str(e)
        })

@app.route('/api/add_group', methods=['POST'])
def add_group():
    if 'user_id' not in session or session.get('position') != 'อาจารย์':
        return jsonify({'success': False, 'message': 'ไม่มีสิทธิ์ในการเพิ่มกลุ่ม'})
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'ไม่พบข้อมูลที่ส่งมา'})
            
        group_name = data.get('name')
        teacher_id = session['user_id']
        
        if not group_name:
            return jsonify({'success': False, 'message': 'กรุณาระบุชื่อกลุ่ม'})
            
        exam_data = load_exam_data()
        if teacher_id not in exam_data['groups']:
            exam_data['groups'][teacher_id] = {}
            
        # Generate new group ID
        existing_ids = [int(id) for id in exam_data['groups'][teacher_id].keys() if id.isdigit()]
        group_id = str(max(existing_ids + [0]) + 1)
            
        exam_data['groups'][teacher_id][group_id] = {
            'name': group_name,
            'tables': {}
        }
        
        save_exam_data(exam_data)
        return jsonify({
            'success': True,
            'group_id': group_id,
            'name': group_name
        })
        
    except Exception as e:
        print(f"Error in add_group: {str(e)}")  # Add debug log
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/api/edit_group', methods=['POST'])
def edit_group():
    if 'user_id' not in session or session.get('position') != 'อาจารย์':
        return jsonify({'success': False, 'message': 'ไม่มีสิทธิ์ในการแก้ไขกลุ่ม'})
        
    try:
        data = request.json
        group_id = data.get('group_id')
        new_name = data.get('name')
        teacher_id = session['user_id']
        
        if not new_name:
            return jsonify({'success': False, 'message': 'กรุณาระบุชื่อกลุ่ม'})
            
        exam_data = load_exam_data()
        if teacher_id not in exam_data['groups'] or group_id not in exam_data['groups'][teacher_id]:
            return jsonify({'success': False, 'message': 'ไม่พบกลุ่มที่ระบุ'})
            
        exam_data['groups'][teacher_id][group_id]['name'] = new_name
        save_exam_data(exam_data)
        
        return jsonify({
            'success': True,
            'group_id': group_id,
            'name': new_name
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/delete_group', methods=['POST'])
def delete_group():
    if 'user_id' not in session or session.get('position') != 'อาจารย์':
        return jsonify({'success': False, 'message': 'ไม่มีสิทธิ์ในการลบกลุ่ม'})
        
    try:
        data = request.json
        group_id = data.get('group_id')
        teacher_id = session['user_id']
        
        exam_data = load_exam_data()
        if teacher_id not in exam_data['groups'] or group_id not in exam_data['groups'][teacher_id]:
            return jsonify({'success': False, 'message': 'ไม่พบกลุ่มที่ระบุ'})
            
        del exam_data['groups'][teacher_id][group_id]
        save_exam_data(exam_data)
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/edit_table', methods=['POST'])
def edit_table():
    if 'user_id' not in session or session.get('position') != 'อาจารย์':
        return jsonify({'success': False, 'message': 'ไม่มีสิทธิ์ในการแก้ไขโต๊ะ'})
        
    try:
        data = request.json
        group_id = data.get('group_id')
        table_id = data.get('table_id')
        new_name = data.get('name')
        teacher_id = session['user_id']
        
        if not new_name:
            return jsonify({'success': False, 'message': 'กรุณาระบุชื่อโต๊ะ'})
            
        exam_data = load_exam_data()
        if (teacher_id not in exam_data['groups'] or 
            group_id not in exam_data['groups'][teacher_id] or
            table_id not in exam_data['groups'][teacher_id][group_id]['tables']):
            return jsonify({'success': False, 'message': 'ไม่พบโต๊ะที่ระบุ'})
            
        exam_data['groups'][teacher_id][group_id]['tables'][table_id]['name'] = new_name
        save_exam_data(exam_data)
        
        return jsonify({
            'success': True,
            'table_id': table_id,
            'name': new_name
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/delete_table', methods=['POST'])
def delete_table():
    if 'user_id' not in session or session.get('position') != 'อาจารย์':
        return jsonify({'success': False, 'message': 'ไม่มีสิทธิ์ในการลบโต๊ะ'})
        
    try:
        data = request.json
        group_id = data.get('group_id')
        table_id = data.get('table_id')
        teacher_id = session['user_id']
        
        exam_data = load_exam_data()
        if (teacher_id not in exam_data['groups'] or 
            group_id not in exam_data['groups'][teacher_id] or
            table_id not in exam_data['groups'][teacher_id][group_id]['tables']):
            return jsonify({'success': False, 'message': 'ไม่พบโต๊ะที่ระบุ'})
            
        # Check if table is occupied
        table = exam_data['groups'][teacher_id][group_id]['tables'][table_id]
        if table.get('status') == 'occupied':
            return jsonify({'success': False, 'message': 'ไม่สามารถลบโต๊ะที่มีนักศึกษาจองแล้ว'})
            
        del exam_data['groups'][teacher_id][group_id]['tables'][table_id]
        save_exam_data(exam_data)
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/add_table', methods=['POST'])
def add_table():
    if 'user_id' not in session or session.get('position') != 'อาจารย์':
        return jsonify({'success': False, 'message': 'ไม่มีสิทธิ์ในการเพิ่มโต๊ะ'})
        
    try:
        data = request.json
        group_id = data.get('group_id')
        table_name = data.get('name', '')  # Optional table name
        teacher_id = session['user_id']
        
        exam_data = load_exam_data()
        if teacher_id not in exam_data['groups'] or group_id not in exam_data['groups'][teacher_id]:
            return jsonify({'success': False, 'message': 'ไม่พบกลุ่มที่ระบุ'})
            
        group = exam_data['groups'][teacher_id][group_id]
        table_id = str(len(group['tables']) + 1)
        
        # If no name provided, use default name
        if not table_name:
            table_name = f'โต๊ะที่ {table_id}'
            
        group['tables'][table_id] = {
            'name': table_name,
            'student_id': None,
            'status': 'available'
        }
        
        save_exam_data(exam_data)
        return jsonify({
            'success': True,
            'table_id': table_id,
            'name': table_name
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/select_seat', methods=['POST'])
def select_seat():
    if 'user_id' not in session or session.get('position') != 'นักศึกษา':
        return jsonify({'success': False, 'message': 'ไม่มีสิทธิ์ในการเลือกที่นั่ง'})
        
    try:
        data = request.json
        teacher_id = data.get('teacher_id')
        group_id = data.get('group_id')
        table_id = data.get('table_id')
        student_id = session['user_id']
        
        exam_data = load_exam_data()
        
        # Check if student already has a seat
        for t_id, teacher_groups in exam_data['groups'].items():
            for g_id, group in teacher_groups.items():
                for tb_id, table in group['tables'].items():
                    if table.get('student_id') == student_id:
                        return jsonify({
                            'success': False, 
                            'message': 'คุณได้จองที่นั่งไว้แล้ว กรุณายกเลิกที่นั่งปัจจุบันก่อนเลือกที่นั่งใหม่'
                        })
        
        if (teacher_id not in exam_data['groups'] or 
            group_id not in exam_data['groups'][teacher_id] or 
            table_id not in exam_data['groups'][teacher_id][group_id]['tables']):
            return jsonify({'success': False, 'message': 'ไม่พบที่นั่งที่ระบุ'})
            
        table = exam_data['groups'][teacher_id][group_id]['tables'][table_id]
        if table['status'] != 'available':
            return jsonify({'success': False, 'message': 'ที่นั่งนี้ถูกจองแล้ว'})
            
        table['student_id'] = student_id
        table['status'] = 'occupied'
        save_exam_data(exam_data)
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/current_seat', methods=['GET'])
def get_current_seat():
    if 'user_id' not in session or session.get('position') != 'นักศึกษา':
        return jsonify({'success': False, 'message': 'ไม่มีสิทธิ์ในการดูข้อมูลที่นั่ง'})
        
    try:
        student_id = session['user_id']
        exam_data = load_exam_data()
        users = load_users()
        
        # Search for student's seat
        for teacher_id, teacher_groups in exam_data['groups'].items():
            for group_id, group in teacher_groups.items():
                for table_id, table in group['tables'].items():
                    if table.get('student_id') == student_id:
                        teacher_name = users[teacher_id]['name']
                        return jsonify({
                            'success': True,
                            'seat': {
                                'teacher_name': teacher_name,
                                'group_name': group['name'],
                                'table_name': table['name']
                            }
                        })
        
        return jsonify({
            'success': True,
            'seat': None
        })
        
    except Exception as e:
        print(f"Error in get_current_seat: {str(e)}")
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/api/student_activity/<group_id>/<table_id>', methods=['GET'])
def get_student_activity(group_id, table_id):
    if 'user_id' not in session or session.get('position') != 'อาจารย์':
        print(f"Unauthorized access attempt - User ID: {session.get('user_id')}, Position: {session.get('position')}")
        return jsonify({'success': False, 'message': 'ไม่มีสิทธิ์ในการดูข้อมูลกิจกรรม'})
        
    try:
        teacher_id = session['user_id']
        print(f"Loading activity data for teacher {teacher_id}, group {group_id}, table {table_id}")
        
        # Load all required data
        exam_data = load_exam_data()
        users = load_users()
        
        # Verify teacher exists
        if teacher_id not in users:
            print(f"Teacher {teacher_id} not found in users database")
            return jsonify({'success': False, 'message': 'ไม่พบข้อมูลอาจารย์ในระบบ'})
        
        # Verify table exists and belongs to teacher
        if teacher_id not in exam_data['groups']:
            print(f"Teacher {teacher_id} has no groups")
            return jsonify({'success': False, 'message': 'ไม่พบข้อมูลกลุ่มของอาจารย์'})
            
        if group_id not in exam_data['groups'][teacher_id]:
            print(f"Group {group_id} not found for teacher {teacher_id}")
            return jsonify({'success': False, 'message': 'ไม่พบข้อมูลกลุ่มที่ระบุ'})
            
        if table_id not in exam_data['groups'][teacher_id][group_id]['tables']:
            print(f"Table {table_id} not found in group {group_id}")
            return jsonify({'success': False, 'message': 'ไม่พบข้อมูลโต๊ะที่ระบุ'})
            
        table = exam_data['groups'][teacher_id][group_id]['tables'][table_id]
        if table['status'] != 'occupied':
            print(f"Table {table_id} is not occupied")
            return jsonify({'success': False, 'message': 'ไม่มีนักศึกษาใช้งานโต๊ะนี้'})
            
        student_id = table['student_id']
        if not student_id:
            print(f"No student assigned to table {table_id}")
            return jsonify({'success': False, 'message': 'ไม่พบข้อมูลนักศึกษาที่โต๊ะนี้'})
            
        if student_id not in users:
            print(f"Student {student_id} not found in users database")
            return jsonify({'success': False, 'message': 'ไม่พบข้อมูลนักศึกษาในระบบ'})
            
        # Get student information
        student = users[student_id]
        print(f"Found student: {student_id} - {student['name']}")
        
        response_data = {
            'success': True,
            'student': {
                'id': student_id,
                'name': student['name'],
                'status': 'active'
            }
        }
        
        print(f"Successfully loaded student data for student {student_id}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in get_student_activity: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/api/request_verification', methods=['POST'])
def request_verification():
    if 'user_id' not in session or session.get('position') != 'อาจารย์':
        return jsonify({'success': False, 'message': 'ไม่มีสิทธิ์ในการขอยืนยันตัวตน'})
        
    try:
        data = request.get_json()
        group_id = data.get('group_id')
        table_id = data.get('table_id')
        
        return jsonify({
            'success': True,
            'message': 'ส่งคำขอยืนยันตัวตนแล้ว'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/api/flag_suspicious', methods=['POST'])
def flag_suspicious():
    if 'user_id' not in session or session.get('position') != 'อาจารย์':
        return jsonify({'success': False, 'message': 'ไม่มีสิทธิ์ในการแจ้งเตือน'})
        
    try:
        data = request.get_json()
        group_id = data.get('group_id')
        table_id = data.get('table_id')
        reason = data.get('reason')
        
        return jsonify({
            'success': True,
            'message': 'บันทึกการแจ้งเตือนแล้ว'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/api/cancel_seat', methods=['POST'])
def cancel_seat():
    if 'user_id' not in session or session.get('position') != 'นักศึกษา':
        return jsonify({'success': False, 'message': 'ไม่มีสิทธิ์ในการยกเลิกที่นั่ง'})
        
    try:
        student_id = session['user_id']
        exam_data = load_exam_data()
        
        # Search for student's seat
        for teacher_id, teacher_groups in exam_data['groups'].items():
            for group_id, group in teacher_groups.items():
                for table_id, table in group['tables'].items():
                    if table.get('student_id') == student_id:
                        # Reset table status
                        table['student_id'] = None
                        table['status'] = 'available'
                        save_exam_data(exam_data)
                        return jsonify({'success': True})
        
        return jsonify({'success': False, 'message': 'ไม่พบที่นั่งที่จองไว้'})
        
    except Exception as e:
        print(f"Error in cancel_seat: {str(e)}")
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/api/cancel_student_seat', methods=['POST'])
def cancel_student_seat():
    if 'user_id' not in session or session.get('position') != 'อาจารย์':
        return jsonify({'success': False, 'message': 'ไม่มีสิทธิ์ในการยกเลิกที่นั่งของนักศึกษา'})
        
    try:
        data = request.json
        group_id = data.get('group_id')
        table_id = data.get('table_id')
        student_id = data.get('student_id')
        teacher_id = session['user_id']
        
        exam_data = load_exam_data()
        
        # Verify table exists and belongs to teacher
        if (teacher_id not in exam_data['groups'] or 
            group_id not in exam_data['groups'][teacher_id] or
            table_id not in exam_data['groups'][teacher_id][group_id]['tables']):
            return jsonify({'success': False, 'message': 'ไม่พบโต๊ะที่ระบุ'})
            
        table = exam_data['groups'][teacher_id][group_id]['tables'][table_id]
        
        # Verify table is occupied by the specified student
        if table.get('student_id') != student_id:
            return jsonify({'success': False, 'message': 'ไม่พบการจองของนักศึกษาที่ระบุ'})
            
        # Reset table status
        table['student_id'] = None
        table['status'] = 'available'
        save_exam_data(exam_data)
        
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"Error in cancel_student_seat: {str(e)}")
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/api/record_website_activity', methods=['POST'])
def record_website_activity():
    if 'user_id' not in session or session.get('position') != 'นักศึกษา':
        return jsonify({'success': False, 'message': 'ไม่มีสิทธิ์ในการบันทึกข้อมูล'})
        
    try:
        data = request.json
        url = data.get('url')
        title = data.get('title')
        student_id = session['user_id']
        
        if not url:
            return jsonify({'success': False, 'message': 'กรุณาระบุ URL'})
            
        monitoring_data = load_monitoring_data()
        
        # Initialize student activities if not exists
        if student_id not in monitoring_data['activities']:
            monitoring_data['activities'][student_id] = []
            
        # Add new activity
        activity = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'url': url,
            'title': title or url
        }
        
        monitoring_data['activities'][student_id].append(activity)
        save_monitoring_data(monitoring_data)
        
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"Error recording website activity: {str(e)}")
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/api/get_student_websites/<student_id>', methods=['GET'])
def get_student_websites(student_id):
    if 'user_id' not in session or session.get('position') != 'อาจารย์':
        return jsonify({'success': False, 'message': 'ไม่มีสิทธิ์ในการดูข้อมูล'})
        
    try:
        monitoring_data = load_monitoring_data()
        student_activities = monitoring_data['activities'].get(student_id, [])
        
        # Sort activities by timestamp in descending order
        student_activities.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Get only the last 50 activities
        recent_activities = student_activities[:50]
        
        return jsonify({
            'success': True,
            'activities': recent_activities
        })
        
    except Exception as e:
        print(f"Error getting student websites: {str(e)}")
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/api/record_resources', methods=['POST'])
def record_resources():
    if 'user_id' not in session or session.get('position') != 'นักศึกษา':
        return jsonify({'success': False, 'message': 'ไม่มีสิทธิ์ในการบันทึกข้อมูล'})
        
    try:
        student_id = session['user_id']
        resource_data = ResourceMonitor.collect_monitoring_data()
        
        monitoring_data = load_monitoring_data()
        
        # Initialize student resources if not exists
        if 'resources' not in monitoring_data:
            monitoring_data['resources'] = {}
        if student_id not in monitoring_data['resources']:
            monitoring_data['resources'][student_id] = []
            
        # Store the monitoring data
        monitoring_data['resources'][student_id].append(resource_data)
        
        # Keep only last 100 records to prevent excessive storage
        if len(monitoring_data['resources'][student_id]) > 100:
            monitoring_data['resources'][student_id] = monitoring_data['resources'][student_id][-100:]
            
        save_monitoring_data(monitoring_data)
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"Error recording resources: {str(e)}")
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/api/get_student_resources/<student_id>', methods=['GET'])
def get_student_resources(student_id):
    if 'user_id' not in session or session.get('position') != 'อาจารย์':
        return jsonify({'success': False, 'message': 'ไม่มีสิทธิ์ในการดูข้อมูล'})
        
    try:
        monitoring_data = load_monitoring_data()
        
        # Get student resources with proper error handling
        student_resources = monitoring_data.get('resources', {}).get(student_id, [])
        
        # Get only the latest resource data
        latest_resource = student_resources[-1] if student_resources else None
        
        if not latest_resource:
            return jsonify({
                'success': True,
                'resources': {
                    'system': {},
                    'browsers': [],
                    'timestamp': None
                }
            })

        # Extract only essential information
        simplified_resource = {
            'system': latest_resource.get('system', {}),
            'browsers': latest_resource.get('browsers', []),
            'timestamp': latest_resource.get('timestamp')
        }
        
        return jsonify({
            'success': True,
            'resources': simplified_resource
        })
        
    except Exception as e:
        print(f"Error getting student resources: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'เกิดข้อผิดพลาดในการโหลดข้อมูล: {str(e)}'
        })

@app.route('/api/verify_student/<group_id>/<table_id>', methods=['POST'])
def verify_student(group_id, table_id):
    if 'user_id' not in session or session.get('position') != 'อาจารย์':
        return jsonify({'success': False, 'message': 'ไม่มีสิทธิ์ในการยืนยันตัวตนนักศึกษา'})
        
    try:
        teacher_id = session['user_id']
        exam_data = load_exam_data()
        
        # Verify table exists and belongs to teacher
        if (teacher_id not in exam_data['groups'] or 
            group_id not in exam_data['groups'][teacher_id] or
            table_id not in exam_data['groups'][teacher_id][group_id]['tables']):
            return jsonify({'success': False, 'message': 'ไม่พบโต๊ะที่ระบุ'})
            
        table = exam_data['groups'][teacher_id][group_id]['tables'][table_id]
        if table['status'] != 'occupied':
            return jsonify({'success': False, 'message': 'ไม่มีนักศึกษาใช้งานโต๊ะนี้'})
            
        student_id = table['student_id']
        if not student_id:
            return jsonify({'success': False, 'message': 'ไม่พบข้อมูลนักศึกษาที่โต๊ะนี้'})
            
        # Update verification status
        if 'verification_status' not in table:
            table['verification_status'] = {}
            
        table['verification_status'][datetime.now().strftime('%Y-%m-%d %H:%M:%S')] = 'verified'
        save_exam_data(exam_data)
        
        return jsonify({
            'success': True,
            'message': 'ยืนยันตัวตนนักศึกษาเรียบร้อยแล้ว'
        })
        
    except Exception as e:
        print(f"Error in verify_student: {str(e)}")
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/monitor/<group_id>/<table_id>')
def monitor_page(group_id, table_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # ตรวจสอบว่าเป็นอาจารย์หรือไม่
    users = load_users()
    user = users.get(session['user_id'])
    if not user or user.get('position') != 'อาจารย์':
        return redirect(url_for('main'))
    
    return render_template('monitor.html')

@app.route('/api/get_resources/<group_id>/<table_id>')
def get_resources(group_id, table_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'กรุณาเข้าสู่ระบบ'})
    
    # ตรวจสอบว่าเป็นอาจารย์หรือไม่
    users = load_users()
    user = users.get(session['user_id'])
    if not user or user.get('position') != 'อาจารย์':
        return jsonify({'success': False, 'message': 'ไม่มีสิทธิ์เข้าถึง'})
    
    try:
        # โหลดข้อมูลการใช้งานทรัพยากร
        monitoring_data = load_monitoring_data()
        resources = monitoring_data.get(group_id, {}).get(table_id, {}).get('resources', {})
        
        return jsonify({
            'success': True,
            'resources': {
                'cpu': resources.get('cpu', 0),
                'memory': resources.get('memory', 0),
                'disk': resources.get('disk', 0)
            }
        })
    except Exception as e:
        print(f"Error getting resources: {str(e)}")
        return jsonify({'success': False, 'message': 'เกิดข้อผิดพลาดในการโหลดข้อมูล'})

@app.route('/api/get_activities/<group_id>/<table_id>')
def get_activities(group_id, table_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'กรุณาเข้าสู่ระบบ'})
    
    # ตรวจสอบว่าเป็นอาจารย์หรือไม่
    users = load_users()
    user = users.get(session['user_id'])
    if not user or user.get('position') != 'อาจารย์':
        return jsonify({'success': False, 'message': 'ไม่มีสิทธิ์เข้าถึง'})
    
    try:
        # โหลดข้อมูลกิจกรรม
        monitoring_data = load_monitoring_data()
        activities = monitoring_data.get(group_id, {}).get(table_id, {}).get('activities', [])
        
        return jsonify({
            'success': True,
            'activities': activities
        })
    except Exception as e:
        print(f"Error getting activities: {str(e)}")
        return jsonify({'success': False, 'message': 'เกิดข้อผิดพลาดในการโหลดข้อมูล'})

@app.route('/monitor_test')
def monitor_test():
    return render_template('monitor_test.html')

@app.route('/static/monitoring_data.json')
def get_monitoring_data():
    try:
        with open('database/monitoring_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        print(f"Error reading monitoring data: {str(e)}")
        return jsonify({
            'resources': {
                '2545': [{
                    'system': {
                        'cpu': {'percent': 0},
                        'memory': {'percent': 0, 'total': 0},
                        'disk': {'percent': 0}
                    },
                    'browsers': [],
                    'active_window': 'ไม่พบข้อมูล',
                    'timestamp': int(time.time())
                }]
            }
        })

@app.route('/api/get_student_details/<group_id>/<table_id>', methods=['GET'])
def get_student_details(group_id, table_id):
    if 'user_id' not in session or session.get('position') != 'อาจารย์':
        return jsonify({'success': False, 'message': 'ไม่มีสิทธิ์ในการดูข้อมูลนักศึกษา'})
        
    try:
        teacher_id = session['user_id']
        print(f"Loading student details for teacher {teacher_id}, group {group_id}, table {table_id}")
        
        # Load required data
        exam_data = load_exam_data()
        users = load_users()
        
        # Verify teacher exists
        if teacher_id not in users:
            print(f"Teacher {teacher_id} not found in users database")
            return jsonify({'success': False, 'message': 'ไม่พบข้อมูลอาจารย์ในระบบ'})
        
        # Verify table exists and belongs to teacher
        if teacher_id not in exam_data['groups']:
            print(f"Teacher {teacher_id} has no groups")
            return jsonify({'success': False, 'message': 'ไม่พบข้อมูลกลุ่มของอาจารย์'})
            
        if group_id not in exam_data['groups'][teacher_id]:
            print(f"Group {group_id} not found for teacher {teacher_id}")
            return jsonify({'success': False, 'message': 'ไม่พบข้อมูลกลุ่มที่ระบุ'})
            
        if table_id not in exam_data['groups'][teacher_id][group_id]['tables']:
            print(f"Table {table_id} not found in group {group_id}")
            return jsonify({'success': False, 'message': 'ไม่พบข้อมูลโต๊ะที่ระบุ'})
            
        table = exam_data['groups'][teacher_id][group_id]['tables'][table_id]
        if table['status'] != 'occupied':
            print(f"Table {table_id} is not occupied")
            return jsonify({'success': False, 'message': 'ไม่มีนักศึกษาใช้งานโต๊ะนี้'})
            
        student_id = table['student_id']
        if not student_id:
            print(f"No student assigned to table {table_id}")
            return jsonify({'success': False, 'message': 'ไม่พบข้อมูลนักศึกษาที่โต๊ะนี้'})
            
        if student_id not in users:
            print(f"Student {student_id} not found in users database")
            return jsonify({'success': False, 'message': 'ไม่พบข้อมูลนักศึกษาในระบบ'})
            
        # Get student information
        student = users[student_id]
        print(f"Found student: {student_id} - {student['name']}")
        
        # Get group information
        group = exam_data['groups'][teacher_id][group_id]
        
        # Prepare response data
        response_data = {
            'success': True,
            'student': {
                'id': student_id,
                'name': student['name'],
                'position': student.get('position', 'นักศึกษา'),
                'face_image': student.get('face_image', '')
            },
            'seat': {
                'group_name': group['name'],
                'table_name': table['name'],
                'status': table['status']
            }
        }
        
        print(f"Successfully loaded student details for student {student_id}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in get_student_details: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/api/allowed_websites', methods=['GET'])
def get_allowed_websites():
    if 'user_id' not in session or session.get('position') != 'อาจารย์':
        return jsonify({'success': False, 'message': 'ไม่มีสิทธิ์ในการดูข้อมูล'})
        
    try:
        exam_data = load_exam_data()
        teacher_id = session['user_id']
        
        # Get websites for this teacher
        websites = exam_data.get('allowed_websites', {}).get(teacher_id, [])
        
        return jsonify({
            'success': True,
            'websites': websites
        })
    except Exception as e:
        print(f"Error getting allowed websites: {str(e)}")
        return jsonify({'success': False, 'message': f'เกิดข้อผิดพลาด: {str(e)}'})

@app.route('/api/add_allowed_website', methods=['POST'])
def add_allowed_website():
    data = request.get_json()
    if not data or 'window_title' not in data:
        return jsonify({'success': False, 'message': 'กรุณากรอก Window Title'})
    
    window_title = data['window_title']
    if not window_title:
        return jsonify({'success': False, 'message': 'กรุณากรอก Window Title'})
    
    try:
        with open('database/exam_data.json', 'r', encoding='utf-8') as f:
            exam_data = json.load(f)
        
        if 'allowed_window_title' not in exam_data:
            exam_data['allowed_window_title'] = {}
        
        user_id = session.get('user_id')
        if user_id not in exam_data['allowed_window_title']:
            exam_data['allowed_window_title'][user_id] = []
        
        if window_title not in exam_data['allowed_window_title'][user_id]:
            exam_data['allowed_window_title'][user_id].append(window_title)
            
            with open('database/exam_data.json', 'w', encoding='utf-8') as f:
                json.dump(exam_data, f, ensure_ascii=False, indent=4)
            
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': 'Window Title นี้มีอยู่ในรายการแล้ว'})
    except Exception as e:
        print(f"Error adding allowed window title: {str(e)}")
        return jsonify({'success': False, 'message': 'เกิดข้อผิดพลาดในการเพิ่ม Window Title'})

@app.route('/api/delete_allowed_website', methods=['POST'])
def delete_allowed_website():
    data = request.get_json()
    if not data or 'window_title' not in data:
        return jsonify({'success': False, 'message': 'กรุณาระบุ Window Title ที่ต้องการลบ'})
    
    window_title = data['window_title']
    try:
        with open('database/exam_data.json', 'r', encoding='utf-8') as f:
            exam_data = json.load(f)
        
        user_id = session.get('user_id')
        if user_id in exam_data.get('allowed_window_title', {}):
            if window_title in exam_data['allowed_window_title'][user_id]:
                exam_data['allowed_window_title'][user_id].remove(window_title)
                
                with open('database/exam_data.json', 'w', encoding='utf-8') as f:
                    json.dump(exam_data, f, ensure_ascii=False, indent=4)
                
                return jsonify({'success': True})
            else:
                return jsonify({'success': False, 'message': 'ไม่พบ Window Title นี้ในรายการ'})
        else:
            return jsonify({'success': False, 'message': 'ไม่พบข้อมูล Window Title ที่อนุญาต'})
    except Exception as e:
        print(f"Error deleting allowed window title: {str(e)}")
        return jsonify({'success': False, 'message': 'เกิดข้อผิดพลาดในการลบ Window Title'})

@app.route('/get_window_status')
def get_window_status():
    try:
        # Get the current user's group from session
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'User not logged in'}), 401

        # Load exam data to get allowed window titles
        exam_data = load_exam_data()
        allowed_titles = exam_data.get('allowed_window_title', {}).get(user_id, [])

        # Load monitoring data to get current active window
        monitoring_data = load_monitoring_data()
        activities = monitoring_data.get('activities', {}).get(user_id, [])
        
        # Get the most recent activity
        if activities:
            latest_activity = activities[-1]
            active_window = latest_activity.get('title', '')
            
            # Check if the active window is in the allowed list
            is_allowed = active_window in allowed_titles
            
            return jsonify({
                'active_window': active_window,
                'is_allowed': is_allowed
            })
        else:
            return jsonify({
                'active_window': 'ไม่พบข้อมูลหน้าต่างที่ใช้งาน',
                'is_allowed': False
            })
            
    except Exception as e:
        print(f"Error in get_window_status: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 