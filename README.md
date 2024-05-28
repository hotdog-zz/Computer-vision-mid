# Computer-vision-mid

本仓库使用mmdetection在VOC数据集上训练并测试目标检测模型Faster R-CNN和YOLO V3。

## 基础设置

1. 从https://github.com/open-mmlab/mmdetection/releases/tag/v3.0.0 下载mmdetection v3.0.0
2. 根据https://mmdetection.readthedocs.io/en/latest/get_started.html 配置虚拟环境，并自行安装tensorboard
3. 修改mmdetection/mmdet/__init__.py中mmcv_maximum_version = '2.3.0'
4. 根据https://mmdetection.readthedocs.io/en/latest/user_guides/dataset_prepare.html 下载并解压VOC2007与VOC2012数据集

## 训练

在config文件夹中包含了两个模型的的config文件，可直接通过mmdetection方式进行训练，可参考如下代码，更多细节参考https://mmdetection.readthedocs.io/en/latest/user_guides/dataset_prepare.html

`
python tools/train.py ${CONFIG_FILE}
`

如果需要开启config中验证集损失函数检测，请将hooks文件夹中文件放入mmdet/engine/hooks内，并取消custom_hooks的注释。

## 测试

将google drive中weight文件夹内权重下载后可直接通过mmdetection方式进行测试，可参考如下代码，更多细节参考https://mmdetection.readthedocs.io/en/latest/user_guides/test.html

`
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
`

仓库也提供了三张demo图片供测试，修改my_test.py文件中相应路径即可实现推理。
