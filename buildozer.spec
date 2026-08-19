[app]
title = TFM 成像
package.name = tfmimaging
package.domain = org.example

source.dir = .
source.include_exts = py,png,jpg,kv,atlas
source.include_patterns = *.py

# 应用版本
version = 1.0

# 依赖（Python 3.12 + kivy + numpy）
# Python 必须 3.12：p4a 默认 3.14 太新，numpy 1.26.4 只支持到 Python 3.12
# numpy 版本号加 v 前缀：master 分支用 git checkout {version}，numpy tag 是 v1.26.4
requirements = python3==3.12.7,hostpython3==3.12.7,kivy==2.3.0,numpy==v1.26.4

# 用 master 分支（源码编译 numpy，不依赖 pip 下载 android wheel）
# develop 分支的 numpy recipe 用 pip 下载 wheel，但 numpy 1.26.4 无 android wheel
p4a.branch = master

# 不打包不需要的
source.exclude_dirs = .git,__pycache__

# 权限：访问共享存储
android.permissions = READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE

# Android 架构（只编 arm64，加快 numpy 源码编译、降低内存压力）
android.archs = arm64-v8a

# 最低/目标 SDK
android.api = 30
android.minapi = 24

# 主程序入口
entrypoint = main

# 横竖屏
orientation = portrait

# 允许后台线程运行
android.allow_backup = True

# 自动接受 Android SDK 许可证（自动化打包必需，否则 build-tools 装不上）
android.accept_sdk_license = True

[buildozer]
log_level = 2
warn_on_root = 1

[android]
# numpy 需要 C 编译，保留默认 recipe 即可
