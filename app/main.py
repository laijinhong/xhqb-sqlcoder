# -*- coding: utf-8 -*-
#
# 全局入口文件
# Author: laijinhong
# Email: laijinhong@xhqb.com
# Created Time: 2024-03-18
# from fastapi import Depends
# from fastapi.middleware.cors import CORSMiddleware
import time
from fastapi import Request
from settings import DEBUG
from utils import parse_readme
from schema import VersionResp
from exceptions import status, InternalException
from init_app import init_app
from llm_module.xhqb_sqlcoder import add_llm_module
from local_file_module.router import router as file_router

# 初始化app
version = "1.0"     # 系统版本号
title, description = parse_readme()
app = init_app(version=version, title=title, description=description, debug=DEBUG)

# 挂载slqcoder模块
add_llm_module(app)

# 加载本地知识文件模块
app.include_router(file_router, prefix="/file", tags=["本地知识文件模块"])

# 跨域问题
"""
origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
"""


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """统一在响应体里注入执行时间的字段"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# redis连接
# from common.connections import init_redis
# init_redis('192.168.1.242')   # 配置redis host

# 加载模块路由
# from test_module.router import router as test_router
# app.include_router(test_router, prefix="/test", tags=["测试模块"])

# 加载验证码模块
# from captcha_module.router import router as captcha_router
# app.include_router(captcha_router, prefix="/captcha", tags=["验证码模块"])
