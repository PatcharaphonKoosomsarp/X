"""
Configuration file for the exam monitoring system
"""

# Face verification settings
FACE_VERIFICATION_INTERVAL = 30  # seconds
FACE_MATCH_TOLERANCE = 0.45  # lower = stricter
MIN_FACE_SIZE_RATIO = 0.2  # minimum face size relative to image

# Browser monitoring settings
BROWSER_CHECK_INTERVAL = 10  # seconds
MAX_ALLOWED_TAB_SWITCHES = 5  # per minute
ALLOWED_DOMAINS = [
    'exam.example.com',  # Replace with actual exam website
    'lms.example.com',   # Replace with actual LMS website
]

# Security settings
BLOCKED_KEYBOARD_SHORTCUTS = [
    'ctrl+c',
    'ctrl+v',
    'ctrl+u',
    'ctrl+s',
    'ctrl+p',
    'ctrl+a',
    'meta+c',
    'meta+v',
    'meta+u',
    'meta+s',
    'meta+p',
    'meta+a'
]

# Notification settings
NOTIFICATION_RETENTION_HOURS = 1
SUSPICIOUS_ACTIVITY_THRESHOLD = 3  # number of suspicious activities before alert

# Database paths
USER_DB_PATH = 'database/users.json'
EXAM_DB_PATH = 'database/exam_data.json'
MONITORING_DB_PATH = 'database/monitoring_data.json'

# Image storage
FACE_IMAGE_PATH = 'new_images'
MAX_IMAGE_SIZE = (640, 480)  # width, height

# Server settings
DEBUG = True
SECRET_KEY = 'your-secret-key-here'  # Change this in production!
HOST = '0.0.0.0'
PORT = 5000

# Logging settings
LOG_FILE = 'logs/monitoring.log'
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' 