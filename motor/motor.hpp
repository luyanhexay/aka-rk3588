#ifndef MOTOR_HPP
#define MOTOR_HPP

#include <memory>
#include <string>
#include "motor_driver.hpp"

// Selects the underlying motor driver implementation.
enum class MotorDriverType {
    PWM,   // sysfs PWM (legacy, 4-channel H-bridge)
    UART,  // ESP32-C3 UART controller
};

// High-level Motor API.
// Interface is identical to aka-sg2002/motor.hpp so tennis.cpp compiles unchanged.
//
// Usage:
//   Motor motor(MotorDriverType::UART, "/dev/ttyS3");
//   motor.forward(50);
class Motor {
public:
    // PWM constructor
    explicit Motor(MotorDriverType type = MotorDriverType::UART,
                   const std::string& device = "/dev/ttyS3");
    ~Motor();

    void forward(int speed);
    void backward(int speed);
    void left(int speed);
    void right(int speed);
    void brake();
    void standby();

    // 差速驱动：正值前进，负值后退，范围 [-100, 100]
    // 非零值会被映射到 [min_speed_, 100]，避免低于死区转不动
    void drive(int left_speed, int right_speed);

    // 设置最小可动速度（默认15），低于此值的非零指令会被拉到该值
    void set_min_speed(int min_speed) { min_speed_ = min_speed; }
    int  get_min_speed() const        { return min_speed_; }

private:
    std::unique_ptr<MotorDriver> driver_;
    int min_speed_ = 15;  // 最小可动速度，低于此值电机可能转不动
};

#endif // MOTOR_HPP
