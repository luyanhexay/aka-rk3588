// Author: Zhihang Shao <dio_ro@outlook.com>
// Source: aka0-ref commits 755a885, 64f8dab, 7b7011d, 9c69f3f, d4fbdad
// Description: ZP10D 舵机 UART 控制实现

#include "arm.hpp"
#include "../logger.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <errno.h>

// 舵机角度实测校准
// 初始(ready): 0=150 1=100 2=110(开)
// 伸下(down):  0=225 1=60  2=110(开)
// 夹住(grab):  0=225 1=60  2=50 (闭)
// 抬起(lift):  0=150 1=100 2=50 (闭) ← 找桶姿势
const float Arm::ID2_ANGLE_OPEN  = 110.0f;
const float Arm::ID2_ANGLE_CLOSE =  50.0f;
const float Arm::ANGLE_MAX       = 270.0f;

const float Arm::SERVO0_READY = 150.0f;
const float Arm::SERVO1_READY = 100.0f;
const float Arm::SERVO0_GRAB  = 225.0f;
const float Arm::SERVO1_GRAB  =  60.0f;
const float Arm::SERVO0_LIFT  = 150.0f;
const float Arm::SERVO1_LIFT  = 100.0f;

Arm::Arm(const std::string& port, int baudrate) : fd_(-1) {
    open_serial(port, baudrate);
}

Arm::~Arm() {
    close_serial();
}

void Arm::open_serial(const std::string& port, int baudrate) {
    fd_ = open(port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd_ < 0) {
        LOGE("[ARM] Failed to open serial port %s: %s", port.c_str(), strerror(errno));
        return;
    }
    // 清除非阻塞标志，确保 write/tcdrain 正常工作
    fcntl(fd_, F_SETFL, 0);

    struct termios tty;
    memset(&tty, 0, sizeof(tty));
    if (tcgetattr(fd_, &tty) != 0) {
        LOGE("[ARM] tcgetattr failed: %s", strerror(errno));
        close_serial();
        return;
    }

    speed_t baud;
    switch (baudrate) {
        case 9600:   baud = B9600;   break;
        case 115200: baud = B115200; break;
        default:     baud = B115200; break;
    }
    cfsetispeed(&tty, baud);
    cfsetospeed(&tty, baud);

    // 8N1, no flow control
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    tty.c_cflag &= ~CRTSCTS;
    tty.c_cflag |= CLOCAL | CREAD;

    // Raw mode
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_iflag &= ~(INLCR | ICRNL | IGNCR);
    tty.c_oflag &= ~OPOST;

    // 100ms timeout
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 1;

    tcflush(fd_, TCIFLUSH);
    if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
        LOGE("[ARM] tcsetattr failed: %s", strerror(errno));
        close_serial();
        return;
    }

    LOGI("[ARM] Serial port %s opened at %d baud", port.c_str(), baudrate);
}

void Arm::close_serial() {
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}

void Arm::send_command(const std::string& cmd) {
    if (fd_ < 0) {
        LOGE("[ARM] Serial port not open, cannot send: %s", cmd.c_str());
        return;
    }
    ssize_t n = write(fd_, cmd.c_str(), cmd.size());
    if (n < 0) {
        LOGE("[ARM] write failed: %s", strerror(errno));
        return;
    }
    tcdrain(fd_);
}

int Arm::angle_to_pulse(float angle) const {
    int pulse = static_cast<int>(500 + (angle / ANGLE_MAX) * 2000);
    return std::max(PULSE_MIN, std::min(PULSE_MAX, pulse));
}

void Arm::set_angle(int servo_id, float angle, int time_ms) {
    if (angle < 0 || angle > ANGLE_MAX) {
        LOGW("[ARM] Angle %.1f out of range [0, %.0f], clamping", angle, ANGLE_MAX);
        angle = std::max(0.0f, std::min(ANGLE_MAX, angle));
    }
    int pulse = angle_to_pulse(angle);
    char buf[32];
    snprintf(buf, sizeof(buf), "#%03dP%04dT%d!", servo_id, pulse, time_ms);
    send_command(buf);
    LOGD("[ARM] servo %d -> angle %.1f (pulse %d, %dms)", servo_id, angle, pulse, time_ms);
}

void Arm::release_torque(int servo_id) {
    char buf[16];
    snprintf(buf, sizeof(buf), "#%03dPULK", servo_id);
    send_command(buf);
    LOGD("[ARM] servo %d torque released", servo_id);
}

void Arm::restore_torque(int servo_id) {
    char buf[16];
    snprintf(buf, sizeof(buf), "#%03dPULR", servo_id);
    send_command(buf);
    LOGD("[ARM] servo %d torque restored", servo_id);
}

// Grab sequence:
// ready(150,100,110开) → down(225,60,110开) → grab(225,60,50闭)
// → lift(150,100,50闭) ← 保持此姿势找桶
void Arm::grab() {
    LOGI("[ARM] Grab sequence start");

    // 1. 伸下去，爪子打开
    set_angle(0, SERVO0_GRAB);
    set_angle(1, SERVO1_GRAB);
    set_angle(2, ID2_ANGLE_OPEN);
    usleep(1500 * 1000);

    // 2. 爪子闭合夹住球
    set_angle(2, ID2_ANGLE_CLOSE);
    usleep(1000 * 1000);

    // 3. 抬起，保持夹住（找桶姿势）
    set_angle(0, SERVO0_LIFT);
    set_angle(1, SERVO1_LIFT);
    usleep(1200 * 1000);

    LOGI("[ARM] Grab sequence done (holding ball, ready to find bin)");
}

void Arm::release_pos() {
    LOGI("[ARM] Moving to release position (down, open)");
    set_angle(0, SERVO0_GRAB);
    set_angle(1, SERVO1_GRAB);
    set_angle(2, ID2_ANGLE_OPEN);
}

void Arm::release() {
    LOGI("[ARM] Releasing ball (open gripper)");
    set_angle(2, ID2_ANGLE_OPEN);
}

void Arm::grab_pos() {
    LOGI("[ARM] Moving to ready position (150,100,110)");
    set_angle(0, SERVO0_READY);
    set_angle(1, SERVO1_READY);
    set_angle(2, ID2_ANGLE_OPEN);
}

void Arm::show() {
    LOGI("[ARM] Lift with ball (150,100,50) - find-bin pose");
    set_angle(0, SERVO0_LIFT);
    set_angle(1, SERVO1_LIFT);
    set_angle(2, ID2_ANGLE_CLOSE);
}
