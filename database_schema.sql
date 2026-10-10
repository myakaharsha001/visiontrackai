-- =====================================================================
-- VisionTrack AI - database schema (MySQL 8.x)
-- Run as a MySQL ADMIN account (e.g. root), NOT as the app user.
-- Edit the password below BEFORE running, and never commit the edited file.
-- =====================================================================

CREATE DATABASE IF NOT EXISTS vision_platform
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

-- Least-privilege application user (change the password!)
CREATE USER IF NOT EXISTS 'vision_user'@'localhost'
    IDENTIFIED BY 'CHANGE_ME_TO_A_STRONG_PASSWORD';

-- Only the rights the app needs on ONLY this database.
-- CREATE is included so the app can run "CREATE TABLE IF NOT EXISTS" itself.
GRANT SELECT, INSERT, CREATE, INDEX ON vision_platform.* TO 'vision_user'@'localhost';
FLUSH PRIVILEGES;

USE vision_platform;

-- Required columns: log_id, timestamp, object_class, confidence, bbox_x/y/w/h.
-- Optional additions (documented in README): source_type, session_id, created_at.
CREATE TABLE IF NOT EXISTS detection_events (
    log_id       BIGINT UNSIGNED  NOT NULL AUTO_INCREMENT,
    `timestamp`  DATETIME(3)      NOT NULL COMMENT 'When the object was detected (app local time)',
    object_class VARCHAR(64)      NOT NULL COMMENT 'YOLO class name, e.g. person',
    confidence   DECIMAL(5,4)     NOT NULL COMMENT 'Model confidence 0.0000 - 1.0000',
    bbox_x       INT              NOT NULL COMMENT 'Left edge (x1), original-frame pixels',
    bbox_y       INT              NOT NULL COMMENT 'Top edge (y1), original-frame pixels',
    bbox_w       INT UNSIGNED     NOT NULL COMMENT 'Width  = x2 - x1, pixels',
    bbox_h       INT UNSIGNED     NOT NULL COMMENT 'Height = y2 - y1, pixels',
    source_type  ENUM('webcam','video_file') NOT NULL DEFAULT 'webcam',
    session_id   CHAR(36)         NULL COMMENT 'UUID of the detection run',
    created_at   TIMESTAMP        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (log_id),
    KEY idx_timestamp (`timestamp`),
    KEY idx_object_class (object_class),
    KEY idx_session (session_id),
    CONSTRAINT chk_confidence CHECK (confidence >= 0 AND confidence <= 1)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
