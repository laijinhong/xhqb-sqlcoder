# -*- coding: utf-8 -*-
#
# 模块路由文件
# Author: laijinhong
# Email: laijinhong@xhqb.com
# Created Time: 2024-03-18
# from typing import Dict
from fastapi import APIRouter
# from fastapi import Depends, HTTPException
from schema import MessageResp     # 通用schema

router = APIRouter(
    # dependencies=[Depends(get_token_header)],
    # responses={404: {"description": "Not found"}},
)


@router.get("/", summary='模块测试API',
            response_model=MessageResp)
async def test_api():
    """模块测试API"""
    return {'message': 'ok'}
