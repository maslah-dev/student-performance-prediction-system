
CREATE DATABASE IF NOT EXISTS student_performance_db;

USE student_performance_db;

-- 1. USERS TABLE
CREATE TABLE IF NOT EXISTS users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(150) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role ENUM('admin', 'educator') NOT NULL DEFAULT 'educator',
    account_status ENUM('active', 'inactive') NOT NULL DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- 2. STUDENTS TABLE
CREATE TABLE IF NOT EXISTS students (
    student_id INT AUTO_INCREMENT PRIMARY KEY,
    registration_number VARCHAR(50) NOT NULL UNIQUE,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    email VARCHAR(150) UNIQUE,
    phone VARCHAR(20),
    gender VARCHAR(20),
    date_of_birth DATE,
    department VARCHAR(100),
    year_of_study INT,
    enrollment_status ENUM('active', 'inactive', 'graduated')
        NOT NULL DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- 3. COURSES TABLE
CREATE TABLE IF NOT EXISTS courses (
    course_id INT AUTO_INCREMENT PRIMARY KEY,
    course_code VARCHAR(30) NOT NULL UNIQUE,
    course_name VARCHAR(150) NOT NULL,
    department VARCHAR(100),
    credit_hours INT,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- 4. ENROLLMENTS TABLE
CREATE TABLE IF NOT EXISTS enrollments (
    enrollment_id INT AUTO_INCREMENT PRIMARY KEY,
    student_id INT NOT NULL,
    course_id INT NOT NULL,
    academic_year VARCHAR(20) NOT NULL,
    semester VARCHAR(30) NOT NULL,
    enrollment_date DATE NOT NULL,
    status ENUM('enrolled', 'completed', 'withdrawn')
        NOT NULL DEFAULT 'enrolled',
    CONSTRAINT fk_enrollment_student FOREIGN KEY (student_id)
        REFERENCES students(student_id),
    CONSTRAINT fk_enrollment_course FOREIGN KEY (course_id)
        REFERENCES courses(course_id),
    CONSTRAINT unique_student_course
        UNIQUE (student_id, course_id, academic_year, semester)
) ENGINE=InnoDB;

-- 5. MARKS TABLE
CREATE TABLE IF NOT EXISTS marks (
    mark_id INT AUTO_INCREMENT PRIMARY KEY,
    enrollment_id INT NOT NULL,
    assessment_name VARCHAR(100) NOT NULL,
    assessment_type ENUM('CAT', 'ASSIGNMENT', 'EXAM', 'QUIZ', 'PROJECT')
        NOT NULL,
    marks_obtained DECIMAL(6,2) NOT NULL,
    total_marks DECIMAL(6,2) NOT NULL,
    assessment_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_marks_enrollment FOREIGN KEY (enrollment_id)
        REFERENCES enrollments(enrollment_id),
    CONSTRAINT chk_marks_obtained CHECK (marks_obtained >= 0),
    CONSTRAINT chk_total_marks CHECK (total_marks > 0),
    CONSTRAINT chk_marks_limit CHECK (marks_obtained <= total_marks)
) ENGINE=InnoDB;

-- 6. ATTENDANCE TABLE
CREATE TABLE IF NOT EXISTS attendance (
    attendance_id INT AUTO_INCREMENT PRIMARY KEY,
    enrollment_id INT NOT NULL,
    attendance_date DATE NOT NULL,
    attendance_status ENUM('present', 'absent', 'late', 'excused') NOT NULL,
    remarks VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_attendance_enrollment FOREIGN KEY (enrollment_id)
        REFERENCES enrollments(enrollment_id),
    CONSTRAINT unique_enrollment_attendance
        UNIQUE (enrollment_id, attendance_date)
) ENGINE=InnoDB;

-- 7. PREDICTIONS TABLE
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INT AUTO_INCREMENT PRIMARY KEY,
    enrollment_id INT NOT NULL,
    predicted_result ENUM('Pass', 'Fail') NOT NULL,
    performance_level ENUM('High', 'Medium', 'Low') NOT NULL,
    failure_probability DECIMAL(5,4) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_prediction_enrollment FOREIGN KEY (enrollment_id)
        REFERENCES enrollments(enrollment_id),
    CONSTRAINT chk_failure_probability
        CHECK (failure_probability BETWEEN 0 AND 1),
    INDEX idx_prediction_enrollment (enrollment_id),
    INDEX idx_prediction_date (prediction_date)
) ENGINE=InnoDB;

-- 8. ALERTS TABLE
CREATE TABLE IF NOT EXISTS alerts (
    alert_id INT AUTO_INCREMENT PRIMARY KEY,
    prediction_id INT NOT NULL,
    alert_type ENUM('Academic Risk', 'Attendance Risk', 'Performance Decline') NOT NULL,
    severity ENUM('Low', 'Medium', 'High', 'Critical') NOT NULL DEFAULT 'Medium',
    alert_message TEXT NOT NULL,
    alert_status ENUM('New', 'Viewed', 'Resolved') NOT NULL DEFAULT 'New',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP NULL,
    CONSTRAINT fk_alert_prediction FOREIGN KEY (prediction_id)
        REFERENCES predictions(prediction_id),
    INDEX idx_alert_status (alert_status)
) ENGINE=InnoDB;

-- 9. AUDIT LOGS TABLE
CREATE TABLE IF NOT EXISTS audit_logs (
    log_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NULL,
    action VARCHAR(100) NOT NULL,
    description TEXT,
    ip_address VARCHAR(45),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_audit_user FOREIGN KEY (user_id)
        REFERENCES users(user_id) ON DELETE SET NULL,
    INDEX idx_audit_user (user_id),
    INDEX idx_audit_date (created_at)
) ENGINE=InnoDB;
