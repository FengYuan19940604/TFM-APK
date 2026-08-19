[app]
title = TFM 成像
package.name = tfmimaging
package.domain = org.example

source.dir = .
source.include_exts = py,png,jpg,kv,atlas
source.include_patterns = *.py

# 应用版本
version = 1.0

# 依赖（kivy + numpy）
requirements = python3,kivy==2.3.0,numpy==1.26.4

# numpy 需要 C 编译，启用 Cython 支持
p4a.branch = master

# 不打包不需要的
source.exclude_dirs = .git,__pycache__

# 权限：访问共享存储
android.permissions = READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE

# Android 架构
android.archs = arm64-v8a,armeabi-v7a

# 最低/目标 SDK
android.api = 30
android.minapi = 24

# 主程序入口
entrypoint = main

# 横竖屏
orientation = portrait

# 允许后台线程运行
android.allow_backup = True
android.accept_sdk_license = True

[buildozer]
log_level = 2
warn_on_root = 1

[android]
# numpy 需要 C 编译，保留默认 recipe 即可
