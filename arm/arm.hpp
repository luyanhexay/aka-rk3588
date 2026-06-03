// Author: Zhihang Shao <dio_ro@outlook.com>
// Source: aka0-ref commits 755a885, 7b7011d, 9c69f3f, d4fbdad
// Description: ZP10D 舵机 UART 控制头文件

#ifndef ARM_HPP
#define ARM_HPP

#include <string>

class Arm {
  public:
    Arm(const std::string& port = "/dev/ttyS2", int baudrate = 115200);
    ~Arm();

    void set_angle(int servo_id, float angle, int time_ms = 1000);
    void release_torque(int servo_id = 255);
    void restore_torque(int servo_id = 255);

    void grab();
    void release();
    void release_pos();
    void grab_pos();  // 收回到待抓取位置（home）
    void show();      // 抬起展示球

  private:
    void open_serial(const std::string& port, int baudrate);
    void close_serial();
    void send_command(const std::string& cmd);
    int angle_to_pulse(float angle) const;

    int fd_;

    static const float ID2_ANGLE_OPEN;
    static const float ID2_ANGLE_CLOSE;
    static const int PULSE_MIN = 500;
    static const int PULSE_MAX = 2500;
    static const float ANGLE_MAX;

    static const float SERVO0_READY;
    static const float SERVO1_READY;
    static const float SERVO0_GRAB;
    static const float SERVO1_GRAB;
    static const float SERVO0_LIFT;
    static const float SERVO1_LIFT;
};

#endif // ARM_HPP
