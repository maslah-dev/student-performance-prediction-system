
USE student_performance_db;

-- 1. SAMPLE STUDENTS
INSERT INTO students
(registration_number, first_name, last_name, email, phone, gender, department, year_of_study)
VALUES
('STU001', 'John', 'Kamau', 'john@example.com', '0700000001', 'Male', 'Computer Science', 2),
('STU002', 'Mary', 'Wanjiku', 'mary@example.com', '0700000002', 'Female', 'Information Technology', 2),
('STU003', 'Peter', 'Otieno', 'peter@example.com', '0700000003', 'Male', 'Computer Science', 3);

-- 2. SAMPLE COURSES
INSERT INTO courses
(course_code, course_name, department, credit_hours)
VALUES
('BIT3105', 'Management Information Systems', 'Information Technology', 3),
('BIT4133', 'Natural Language Processing', 'Computer Science', 3),
('BIT3106', 'Machine Learning', 'Computer Science', 3);

-- 3. SAMPLE ENROLLMENTS
INSERT INTO enrollments
(student_id, course_id, academic_year, semester, enrollment_date)
VALUES
(1, 1, '2026/2027', 'Semester 1', '2026-11-01'),
(1, 3, '2026/2027', 'Semester 1', '2026-11-01'),
(2, 1, '2026/2027', 'Semester 1', '2026-11-01'),
(2, 2, '2026/2027', 'Semester 1', '2026-11-01'),
(3, 2, '2026/2027', 'Semester 1', '2026-11-01'),
(3, 3, '2026/2027', 'Semester 1', '2026-11-01');

-- 4. SAMPLE MARKS
INSERT INTO marks
(enrollment_id, assessment_name, assessment_type, marks_obtained, total_marks, assessment_date)
VALUES
(1, 'CAT 1', 'CAT', 18, 20, '2026-11-05'),
(1, 'Assignment 1', 'ASSIGNMENT', 15, 20, '2026-11-10'),
(1, 'Final Exam', 'EXAM', 65, 100, '2026-11-17'),

(2, 'CAT 1', 'CAT', 12, 20, '2026-11-05'),
(2, 'Assignment 1', 'ASSIGNMENT', 14, 20, '2026-11-10'),
(2, 'Final Exam', 'EXAM', 45, 100, '2026-11-17'),

(3, 'CAT 1', 'CAT', 19, 20, '2026-11-05'),
(3, 'Assignment 1', 'ASSIGNMENT', 18, 20, '2026-11-10'),
(3, 'Final Exam', 'EXAM', 85, 100, '2026-11-17'),

(4, 'CAT 1', 'CAT', 16, 20, '2026-11-05'),
(4, 'Assignment 1', 'ASSIGNMENT', 17, 20, '2026-11-10'),
(4, 'Final Exam', 'EXAM', 72, 100, '2026-11-17'),

(5, 'CAT 1', 'CAT', 8, 20, '2026-11-05'),
(5, 'Assignment 1', 'ASSIGNMENT', 10, 20, '2026-11-10'),
(5, 'Final Exam', 'EXAM', 38, 100, '2026-11-17'),

(6, 'CAT 1', 'CAT', 11, 20, '2026-11-05'),
(6, 'Assignment 1', 'ASSIGNMENT', 12, 20, '2026-11-10'),
(6, 'Final Exam', 'EXAM', 42, 100, '2026-11-17');

-- 5. SAMPLE ATTENDANCE
INSERT INTO attendance
(enrollment_id, attendance_date, attendance_status)
VALUES
(1, '2026-11-03', 'present'),
(1, '2026-11-05', 'present'),
(1, '2026-11-10', 'present'),
(1, '2026-11-12', 'absent'),
(1, '2026-11-17', 'present'),

(2, '2026-11-03', 'present'),
(2, '2026-11-05', 'absent'),
(2, '2026-11-10', 'late'),
(2, '2026-11-12', 'absent'),
(2, '2026-11-17', 'present'),

(3, '2026-11-03', 'present'),
(3, '2026-11-05', 'present'),
(3, '2026-11-10', 'present'),
(3, '2026-11-12', 'present'),
(3, '2026-11-17', 'present'),

(4, '2026-11-03', 'present'),
(4, '2026-11-05', 'present'),
(4, '2026-11-10', 'present'),
(4, '2026-11-12', 'present'),
(4, '2026-11-17', 'late'),

(5, '2026-11-03', 'absent'),
(5, '2026-11-05', 'present'),
(5, '2026-11-10', 'absent'),
(5, '2026-11-12', 'absent'),
(5, '2026-11-17', 'present'),

(6, '2026-11-03', 'present'),
(6, '2026-11-05', 'absent'),
(6, '2026-11-10', 'absent'),
(6, '2026-11-12', 'present'),
(6, '2026-11-17', 'absent');

-- 6. SAMPLE PREDICTIONS
-- These are demonstration records, not trained-model predictions.
INSERT INTO predictions
(enrollment_id, predicted_result, performance_level, failure_probability, model_name, model_version)
VALUES
(1, 'Pass', 'High', 0.1000, 'Random Forest', '1.0'),
(2, 'Fail', 'Low', 0.7500, 'Random Forest', '1.0'),
(3, 'Pass', 'High', 0.0500, 'Random Forest', '1.0'),
(4, 'Pass', 'High', 0.1200, 'Random Forest', '1.0'),
(5, 'Fail', 'Low', 0.8500, 'Random Forest', '1.0'),
(6, 'Fail', 'Low', 0.7000, 'Random Forest', '1.0');

-- 7. SAMPLE ALERTS
INSERT INTO alerts
(prediction_id, alert_type, severity, alert_message)
VALUES
(2, 'Academic Risk', 'High',
 'John Kamau has a 75% predicted failure probability in BIT3106. Academic support is recommended.'),
(5, 'Academic Risk', 'Critical',
 'Peter Otieno has an 85% predicted failure probability in BIT4133. Immediate academic intervention is recommended.'),
(6, 'Academic Risk', 'High',
 'Peter Otieno has a 70% predicted failure probability in BIT3106. Academic support is recommended.');

-- 8. SAMPLE USERS
-- Password hashes here are placeholders, not usable login credentials.
INSERT INTO users
(full_name, email, password_hash, role, account_status)
VALUES
('System Administrator', 'admin@example.com',
 'DEMO_HASH_REPLACE_BEFORE_LOGIN', 'admin', 'active'),
('Sample Educator', 'educator@example.com',
 'DEMO_HASH_REPLACE_BEFORE_LOGIN', 'educator', 'active');

-- 9. SAMPLE AUDIT LOGS
INSERT INTO audit_logs
(user_id, action, description, ip_address)
VALUES
(1, 'SYSTEM_SETUP',
 'Administrator account created during initial system setup.',
 '127.0.0.1'),
(2, 'SYSTEM_SETUP',
 'Educator account created during initial system setup.',
 '127.0.0.1'),
(1, 'DATABASE_SETUP',
 'Sample students, courses, marks, attendance, predictions and alerts added.',
 '127.0.0.1');
