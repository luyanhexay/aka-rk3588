#pragma once
// Sub-commands for debugging: test-uvc, test-yolo, test-motor

int cmd_test_uvc(int uvc_index);
int cmd_test_yolo(const char* model_path, int uvc_index);
int cmd_test_motor(const char* uart_dev, int argc, char** argv);
