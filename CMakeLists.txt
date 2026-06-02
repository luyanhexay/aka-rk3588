cmake_minimum_required(VERSION 3.10)
project(aka_rk3588_tennis)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Log level ─────────────────────────────────────────────────────────────────
if(NOT DEFINED LOG_LEVEL)
  set(LOG_LEVEL "INFO")
endif()
if(LOG_LEVEL STREQUAL "DEBUG")
  add_definitions(-DCURRENT_LOG_LEVEL=0)
elseif(LOG_LEVEL STREQUAL "INFO")
  add_definitions(-DCURRENT_LOG_LEVEL=1)
elseif(LOG_LEVEL STREQUAL "WARN")
  add_definitions(-DCURRENT_LOG_LEVEL=2)
elseif(LOG_LEVEL STREQUAL "ERROR")
  add_definitions(-DCURRENT_LOG_LEVEL=3)
else()
  add_definitions(-DCURRENT_LOG_LEVEL=1)
endif()

# ── 3rd: self-contained dependency tree (no ../yolov8 needed) ─────────────────
set(3RD ${CMAKE_CURRENT_SOURCE_DIR}/3rd)

if(NOT DEFINED TARGET_SOC)
  set(TARGET_SOC rk3588)
endif()

# ── NATIVE_BUILD: compile directly on the board (no cross-toolchain needed) ──
option(NATIVE_BUILD "Build natively on the board using system libraries" OFF)

# ── RKNN YOLOv8 sources (rknpu2 for rk3588) ──────────────────────────────────
set(RKNPU2_SRC
    ${3RD}/yolov8_src/postprocess.cc
    ${3RD}/yolov8_src/rknpu2/yolov8.cc
)

# ── Common 3rd paths ──────────────────────────────────────────────────────────
set(RKNNRT_DIR         ${3RD}/rknpu2)
set(LIBRKNNRT          ${RKNNRT_DIR}/Linux/aarch64/librknnrt.so)
set(LIBRKNNRT_INCLUDES ${RKNNRT_DIR}/include)

set(STB_INCLUDES        ${3RD}/stb_image)
set(LIBRGA_INCLUDES     ${3RD}/librga/include)

# tennis.rknn is a single-class model (tennis ball only): output=[1,5,N]
add_definitions(-DOBJ_CLASS_NUM=1)
add_definitions(-DDISABLE_RGA)

if(NATIVE_BUILD)
  # ── Board native build ─────────────────────────────────────────────────────
  # librknnrt is already installed on the board; link dynamically, no .so in repo.
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(LIBUVC    REQUIRED libuvc)
  pkg_check_modules(LIBUSB    REQUIRED libusb-1.0)
  pkg_check_modules(LIBJPEG   REQUIRED libturbojpeg)
  pkg_check_modules(LIBSNDFILE REQUIRED sndfile)

  set(LIBJPEG_INCLUDES    ${LIBJPEG_INCLUDE_DIRS})
  set(LIBJPEG             ${LIBJPEG_LIBRARIES})
  set(LIBRGA              "")
  set(LIBSNDFILE          ${LIBSNDFILE_LIBRARIES})
  set(LIBSNDFILE_INCLUDES ${LIBSNDFILE_INCLUDE_DIRS})

  add_subdirectory(${3RD}/utils utils.out)

  include_directories(
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${LIBRKNNRT_INCLUDES}
      ${3RD}/yolov8_src
      ${3RD}/utils
      ${LIBUVC_INCLUDE_DIRS}
      ${LIBUSB_INCLUDE_DIRS}
      ${LIBJPEG_INCLUDE_DIRS}
  )

  add_executable(tennis
      tennis.cpp
      motor/motor.cpp motor/pwm_motor_driver.cpp motor/uart_motor_driver.cpp
      capture/uvc_capture.cpp
      detect/detect.cpp
      ${RKNPU2_SRC}
  )

  # Use the bundled librknnrt.so (newer than what's installed on the board)
  target_link_libraries(tennis
      imageutils fileutils imagedrawing
      ${LIBRKNNRT}
      ${LIBUVC_LIBRARIES}
      ${LIBUSB_LIBRARIES}
      ${LIBJPEG_LIBRARIES}
      -lpthread -ldl -lm
  )

  # Set RPATH so the binary finds the bundled librknnrt.so at runtime.
  # Deploy the .so alongside the binary (e.g. same directory or ../lib/).
  set_target_properties(tennis PROPERTIES
      BUILD_RPATH   "${RKNNRT_DIR}/Linux/aarch64"
      INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
  )

else()
  # ── Cross-compile on PC ────────────────────────────────────────────────────
  set(LIBJPEG_DIR       ${3RD}/jpeg_turbo)
  set(LIBJPEG_INCLUDES  ${LIBJPEG_DIR}/include)
  set(LIBJPEG           ${LIBJPEG_DIR}/Linux/aarch64/libturbojpeg.a)
  set(LIBRGA            ${3RD}/librga/Linux/aarch64/librga.so)
  set(LIBSNDFILE        "")
  set(LIBSNDFILE_INCLUDES "")

  add_subdirectory(${3RD}/utils utils.out)

  include_directories(
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${LIBRKNNRT_INCLUDES}
      ${3RD}/yolov8_src
      ${3RD}/utils
      ${LIBJPEG_DIR}/include
      /usr/include   # fallback for libuvc/libuvc.h
  )

  add_executable(tennis
      tennis.cpp
      motor/motor.cpp motor/pwm_motor_driver.cpp motor/uart_motor_driver.cpp
      capture/uvc_capture.cpp
      detect/detect.cpp
      ${RKNPU2_SRC}
  )

  # Generate empty stub .so so the linker can resolve -luvc etc.
  foreach(STUBLIB uvc usb-1.0 turbojpeg)
      set(STUB_SRC ${CMAKE_CURRENT_BINARY_DIR}/stub_${STUBLIB}.c)
      set(STUB_SO  ${CMAKE_CURRENT_BINARY_DIR}/lib${STUBLIB}.so)
      file(WRITE ${STUB_SRC} "/* stub */\n")
      add_custom_command(
          OUTPUT  ${STUB_SO}
          COMMAND ${CMAKE_C_COMPILER} -shared -fPIC -o ${STUB_SO} ${STUB_SRC}
          DEPENDS ${STUB_SRC}
          COMMENT "Generating stub lib${STUBLIB}.so"
      )
      list(APPEND STUB_SO_LIST ${STUB_SO})
  endforeach()
  add_custom_target(stub_libs ALL DEPENDS ${STUB_SO_LIST})
  add_dependencies(tennis stub_libs)

  target_link_options(tennis PRIVATE
      -Wl,--allow-shlib-undefined
      -Wl,--unresolved-symbols=ignore-all
      -L${CMAKE_CURRENT_BINARY_DIR}
  )

  target_link_libraries(tennis
      imageutils fileutils imagedrawing
      ${LIBRKNNRT}
      -Wl,--no-as-needed
      -luvc -lusb-1.0 -lturbojpeg
      -Wl,--as-needed
      -lpthread -ldl -lm
  )
endif()

set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
install(TARGETS tennis DESTINATION bin)
