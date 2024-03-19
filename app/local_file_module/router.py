# -*- coding: utf-8 -*-
#
# 模块路由文件
# Author: laijinhong
# Email: laijinhong@xhqb.com
# Created Time: 2024-03-18
# from typing import Dict
import os.path
import sys
import re

from fastapi import APIRouter, UploadFile, File
from fastapi.exceptions import RequestValidationError
from typing import List

# from fastapi import Depends, HTTPException
sys.path.append("..")
from settings import XHQB_LOCAL_FILE_PATH

router = APIRouter(
    # dependencies=[Depends(get_token_header)],
    # responses={404: {"description": "Not found"}},
)


@router.post("/upload/", summary="上传本地知识文件", include_in_schema=False)
async def upload_file(file: UploadFile = File(...)):
    file_location = f"{XHQB_LOCAL_FILE_PATH}/{file.filename}"
    if os.path.isfile(file_location):
        raise RequestValidationError(errors=[f"文件名{file.filename}已存在"])
    match = re.search(r".*\.txt", file.filename)
    if not match:
        raise RequestValidationError(errors=["只支持txt文件格式"])
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())


@router.post("/list/", summary="查询文件列表", response_model=List[str])
async def upload_file():
    files = [f for f in os.listdir(XHQB_LOCAL_FILE_PATH) if os.path.isfile(XHQB_LOCAL_FILE_PATH + "/" + f)]
    return files
