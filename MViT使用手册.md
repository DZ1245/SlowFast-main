# MViT使用手册

* 代码使用来自Facebook的SlowFast仓库

## 环境配置

* python>=3.8.0 pytorch>=1.12.0 
* torchvison默认版本过高，导致无法使用其进行解码
* detection2安装注意对应cuda版本

## Config设置

* CHECKPOINT_FILE_PATH：存放checkpoint权重或者预训练权重的地址。
* DECODING_BACKEND：解码视频所使用库，pyav或者torchvison，torchvision因版本更新改变了所包含的函数导致无法使用。
* PATH_TO_DATA_DIR：train.csv和val.csv的存放路径。eg:/home/dz/workspace/SlowFast-main/datapath/kinetics400/
* PATH_PREFIX：视频数据的存放路径，和csv中的路径结合可以读取视频。eg:/data/home/dz/Kinetics400/
* PATH_LABEL_SEPARATOR：csv中数据的分割符。eg:','
* OUTPUT_DIR：checkpoint和json的输出位置。eg:/data/home/dz/checkpoint_log/SlowFast_output/Test_MViT2_S_400zero

## 代码修改

* pyav解码过程存在缺陷，会出现int和list类型错误:

  ```python
  pyav with exception: unsupported operand type(s) for -: 'list' and 'int'
  ```

  解决方法是对decoder中pyav部分代码进行修改，如以下代码等：

  ```python
  if duration is None:
          # If failed to fetch the decoding information, decode the entire video.
          decode_all_video = True
          video_start_pts, video_end_pts = 0, math.inf
      else:
          # Perform selective decoding.
          decode_all_video = False
  
          clip_sizes = [
              np.maximum(
                  1.0,
                  np.ceil(
                      sampling_rate[i] * (num_frames[i] - 1) / target_fps * fps
                  ),
              )
              for i in range(len(sampling_rate))
          ]
  ```

  参考：

  [Torchvision backend not working. · Issue #181 · facebookresearch/SlowFast (github.com)](https://github.com/facebookresearch/SlowFast/issues/181)

  [PyAV decoding backend no longer works · Issue #563 · facebookresearch/SlowFast (github.com)](https://github.com/facebookresearch/SlowFast/issues/563)

  [Fix decoding issue with PYAV due to new support for multiple training… by dfan · Pull Request #541 · facebookresearch/SlowFast (github.com)](https://github.com/facebookresearch/SlowFast/pull/541)

* 训练过程中出现DataLoader worker is killed问题，错误在于woker设置过大，修改config中的设置。

* 在使用预训练模型进行继续训练或微调时候，会出现Keyerror，显示在param_group中没有"layer_decay"：

  ```python
  for param_group in optimizer.param_groups:
          param_group["lr"] = new_lr * param_group["layer_decay"]
  ```

  问题在于从权重中导入预训练数据时，并不包含optimizer的数据，所以会出现Keyerror，解决方法是手动设置layer_decay这一参数，于train_net中设置：
  
  ```python
  # 第599行
  optimizer.param_groups = [{**x, **{'layer_decay': 1.0}} for x in optimizer.param_groups]
  ```
  
  此代码的各部分含义如下：
  
  1. `optimizer.param_groups`：这是一个包含优化器中所有参数组的列表。每个参数组都是一个字典，包含一部分模型参数的参数和对应的优化选项。
  2. `[ {**x, **{'layer_decay': 1.0}} for x in optimizer.param_groups]`：这是一个列表推导式，通过迭代原始参数组列表并创建一个新的字典，其中包含原始字典相同的键和值，但还包含一个额外的键值对，即 `layer_decay` 参数的值为 1.0。 `**` 语法用于将原始字典和新的键值对展开为一个字典。
  3. `optimizer.param_groups = ...`：这将新的参数组列表赋值给优化器的 `param_groups` 属性，有效地更新了优化器中所有参数组的 `layer_decay` 参数的值。
  
  总之，这段代码将优化器中所有参数组的 `layer_decay` 参数设置为 1.0，可以在训练神经网络时应用层级的权重衰减正则化。

## 数据集问题

* 使用初始的数据集时，train集解码正常可以进行训练，但是val集.mkv和.webb等后缀的文件基本无法解码，解码出的数据中'duration' param is None。解决方法是从别的途径获取数据集，参考：

  [pyav decode leads memory leaks issue when 'duration' is none · Issue #626 · facebookresearch/SlowFast (github.com)](https://github.com/facebookresearch/SlowFast/issues/626)

  [Datasets-OpenDataLab](https://opendatalab.com/)