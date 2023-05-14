<center><br>基于文档的聊天机器人</br></center>


#### 设计思路

* 基于本地文档回答内容
* 引入了记忆模式，用户提问相似的内容回到历史记录中查找，减少token的消耗
* 形式暂定为命令行模式，后续加入web-UI
  

#### 开发进展
* 完成了最初的CLI demo.
* 成功引入了基于文档的机制
* 引入了记忆模式
* 目前仅支持html文档


#### 脚本用法Usage

* python 版本3.11.3：

```shell
conda create -n env_name --y
pip install -r requirement.txt

```

* 使用
```bash

python demo.py --query [问题] --file [文件夹｜文件]

```

#### ToDo

* Prompt模版还需修改，直到回答内容尽量为中文
* 构建本地向量数据库
* 构建问答对
* 统一不同格式文档的入口
* 自动检测输入数据类型
