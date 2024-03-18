# -*- coding: utf-8 -*-
#
# 工具库
# Author: laijinhong
# Email: laijinhong@xhqb.com
# Created Time: 2024-03-18


def parse_readme(filename: str = 'readme.md'):
    """解释readme文件
    :param filename str: md文件名
    :return title str: 标题
    :return text  str: 文档内容
    """
    with open(filename, encoding='utf8') as r:
        text = r.readlines()
    title = text[0].strip('# \n').strip()
    text = ''.join(text[1:]).strip()
    return title, text
