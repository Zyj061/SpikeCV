基于脉冲神经网络的多目标跟踪SNN_Tracker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``spkProc.tracking.SNN_Tracker.snn_tracker``\ 中包含基于脉冲神经网络（Spiking Neural Network, SNN）的多目标跟踪算法\ ``SNNTracker``\ 。其核心算法思想是通过结合STP滤波器、动态神经场（DNF）、STDP聚类和运动估计等多个模块，实现面向脉冲相机的多目标跟踪。SNN_Tracker的跟踪框架如下图所示：

.. image:: ./media/snn_tracker.png
   :target: ./media/snn_tracker.png
   :alt: snn_tracker

SNN_Tracker框架包含以下核心模块：

* ``动态适应层``\ ：STP脉冲滤波器，用于过滤出脉冲流中的运动物体
* ``检测层``\ ：基于动态神经场（DNF）的运动物体检测器，用于找到不同的运动物体
* ``聚类层``\ ：基于STDP的物体聚类模块，用于将检测到的运动物体进行聚类和跟踪
* ``运动估计层``\ ：基于STDP的运动估计模块，用于估计物体的运动向量

使用SNN_Tracker算法可先通过实例化\ ``spkProc.tracking.SNN_Tracker.snn_tracker``\ 中的\ ``SNNTracker``\ 类，其采用的数据类型为 *pytorch* 的张量形式，初始化时需提供脉冲阵列的高度\ ``spike_h``\ ，宽度\ ``spike_w``\ ，处理器\ ``device``\ ，以及可选的注意力区域大小\ ``attention_size``\ 和差分时间\ ``diff_time``\ 。例如，通过以下例子进行创建跟踪实例：

.. code-block:: python

   from spkProc.tracking.SNN_Tracker.snn_tracker import SNNTracker
   import torch

   device = torch.device('cuda')
   # spikes为使用SpikeStream实例获得的脉冲流矩阵
   spike_tracker = SNNTracker(spike_h=250, spike_w=400, device=device, attention_size=20)

SNNTracker类中的变量
~~~~~~~~~~~~~~~~~~~~~

``snn_tracker.py``\ 中SNN跟踪器对应的类\ ``SNNTracker``\ 具有以下几种变量：

* ``spike_h``\ ：脉冲阵列高度
* ``spike_w``\ ：脉冲阵列宽度
* ``device``\ ：所使用的处理器类型，\ ``cpu``\ 或者\ ``cuda``
* ``attention_size``\ ：搜索区域半径，默认为20
* ``diff_time``\ ：差分时间，用于STP滤波器的时间差分，默认为1
* ``stp_filter``\ ：STP脉冲滤波器，\ ``spkProc.filters.stp_filters_torch.STPFilter``\ 类的实例，对应动态适应层
* ``object_detection``\ ：运动物体检测器，\ ``spkProc.detection.attention_select.SaccadeInput``\ 类的实例，对应检测层
* ``motion_estimator``\ ：运动估计器，\ ``spkProc.motion.motion_detection.motion_estimation``\ 类的实例，对应运动估计层
* ``object_cluster``\ ：物体聚类器，\ ``spkProc.detection.stdp_clustering.stdp_cluster``\ 类的实例，对应聚类层
* ``calibration_time``\ ：校正时间步，在开始对运动物体进行检测跟踪前，运行滤波器以滤除冗余脉冲的步长，默认为150
* ``timestamps``\ ：当前处理的时间戳计数
* ``trajectories``\ ：存储跟踪轨迹的字典，键为跟踪ID，值为\ ``trajectories``\ 命名元组
* ``filterd_spikes``\ ：保留滤波器滤除后的脉冲流，可用于导出可视化的跟踪结果

SNNTracker类中的函数
~~~~~~~~~~~~~~~~~~~~~

``snn_tracker.SNNTracker``\ 中包含以下五个函数：

#. 
   ``calibrate_motion(spikes, calibration_time=None)``\ ：在开始跟踪前运行滤波器以滤除冗余脉冲，进行运动校准。该方法会根据指定的校正时长（或使用默认值）更新STP滤波器的状态。

   其调用方式如下：

   .. code-block:: python

      spike_tracker = SNNTracker(spike_h=250, spike_w=400, device=torch.device('cuda'))
      # spikes 是通过spkData.load_dat.SpikeStream对象获得的脉冲流
      spike_tracker.calibrate_motion(spikes, calibration_time=150)

#. 
   ``get_results(spikes, res_filepath, mov_writer=None, save_video=False)``\ ：执行多目标检测跟踪，并将结果保存至\ ``res_filepath``\ 中指定的 *txt* 文件中。该方法会遍历所有时间步，依次执行STP滤波、DNF检测、STDP聚类和运动估计，最终输出跟踪结果。

   参数说明：

   * ``spikes``\ ：脉冲流矩阵，形状为\ ``(timesteps, spike_h, spike_w)\``
   * ``res_filepath``\ ：跟踪结果保存路径
   * ``mov_writer``\ ：可选的视频写入器，用于保存可视化结果
   * ``save_video``\ ：是否保存可视化视频，默认为\ ``False``

   其调用方式如下：

   .. code-block:: python

      spike_tracker = SNNTracker(spike_h=250, spike_w=400, device=torch.device('cuda'))
      spike_tracker.calibrate_motion(spikes, calibration_time=150)
      spike_tracker.get_results(spikes, 'results/tracking_result.txt', save_video=True)

#. 
   ``_plot_timing_curve(timing_data, res_filepath)``\ ：绘制STDP跟踪的耗时曲线图，并将结果保存为PNG图片和CSV文件。该方法用于分析跟踪算法的性能。

   参数说明：

   * ``timing_data``\ ：耗时数据列表，单位为毫秒
   * ``res_filepath``\ ：结果文件路径，用于生成图片文件名

   该方法会自动生成以下文件：

   * ``{base_name}_stdp_timing_curve.png``\ ：耗时曲线图
   * ``{base_name}_stdp_timing_data.csv``\ ：耗时数据CSV文件

#. 
   ``save_trajectory(results_dir, data_name)``\ ：保存跟踪轨迹到JSON文件。该方法会将所有跟踪对象的轨迹和边界框信息保存为JSON格式。

   参数说明：

   * ``results_dir``\ ：结果保存目录
   * ``data_name``\ ：数据集名称，用于生成文件名

   该方法会生成以下文件：

   * ``{data_name}_py.json``\ ：Python格式的轨迹JSON文件
   * ``{data_name}.json``\ ：MATLAB格式的轨迹JSON文件
   * ``{data_name}_bbox.json``\ ：边界框JSON文件

#. 
   ``__init__(spike_h, spike_w, device, attention_size=20, diff_time=1, **STPargs)``\ ：初始化SNN跟踪器，创建所有必要的组件。

   参数说明：

   * ``spike_h``\ ：脉冲阵列高度
   * ``spike_w``\ ：脉冲阵列宽度
   * ``device``\ ：处理器类型，\ ``cpu``\ 或者\ ``cuda``
   * ``attention_size``\ ：搜索区域半径，默认为20
   * ``diff_time``\ ：差分时间，默认为1
   * ``**STPargs``\ ：STP滤波器的额外参数，包括\ ``u0``\ 、\ ``D``\ 、\ ``F``\ 、\ ``f``\ 、\ ``time_unit``\ 等

   其调用方式如下：

   .. code-block:: python

      from spkProc.tracking.SNN_Tracker.snn_tracker import SNNTracker
      import torch

      device = torch.device('cuda')

      # 使用默认STP参数
      spike_tracker = SNNTracker(spike_h=250, spike_w=400, device=device)

      # 自定义STP参数
      stpPara = {
          'u0': 0.15,
          'D': 0.05 * 20,
          'F': 0.5 * 20,
          'f': 0.15,
          'time_unit': 1
      }
      spike_tracker = SNNTracker(spike_h=250, spike_w=400, device=device, **stpPara)

使用示例
~~~~~~~~

下面是一个完整的使用示例，展示如何使用SNNTracker进行多目标跟踪：

.. code-block:: python

   import torch
   from spkData.load_dat_snntrack import SpikeStream
   from spkProc.tracking.SNN_Tracker.snn_tracker import SNNTracker

   # 加载数据
   dataDict = SpikeStream.data_parameter_dict(
       'motVidarReal2020/spike59',
       spike_h=250,
       spike_w=400,
       labeled_data_type='tracking'
   )

   # 创建脉冲流对象
   spike_stream = SpikeStream(**dataDict)
   spikes = spike_stream.spikes

   # 创建SNN跟踪器
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   spike_tracker = SNNTracker(
       spike_h=250,
       spike_w=400,
       device=device,
       attention_size=20
   )

   # 校准运动
   calibration_time = 150
   spike_tracker.calibrate_motion(spikes, calibration_time=calibration_time)

   # 执行跟踪
   tracking_file = 'results/tracking_result.txt'
   spike_tracker.get_results(
       spikes[calibration_time:],
       tracking_file,
       save_video=True
   )

   # 保存轨迹
   spike_tracker.save_trajectory('results', 'spike59')

性能分析
~~~~~~~~

SNNTracker内置了性能分析功能，会自动记录STDP跟踪模块的耗时信息，并在跟踪完成后输出统计信息：

.. code-block:: text

   stdp_tracking avg/min/max over 1000 frames: 2.345 / 1.234 / 5.678 ms
   Total tracking took: 123.456 seconds for 1000 timestamps spikes
   Timing curve plot saved to: results/tracking_result_stdp_timing_curve.png
   Timing data saved to: results/tracking_result_stdp_timing_data.csv

生成的耗时曲线图包含以下信息：

* 每次调用的耗时曲线
* 平均耗时（绿色虚线）
* 最小耗时（蓝色点线）
* 最大耗时（红色点线）

CSV文件包含详细的耗时数据，可用于进一步分析。

.. note::
   SNNTracker在运行过程中会自动处理GPU内存不足的情况，如果遇到CUDA OOM错误，会自动清理缓存并继续运行。

相关论文
~~~~~~~~

更多关于SNN_Tracker多目标跟踪算法的细节可参考论文：

#. Huang T, Zheng Y, Yu Z, et al. 1000× Faster Camera and Machine Vision with Ordinary Devices[J]. Engineering, 2022.
#. Zheng Y, Zheng L, Yu Z, et al. High-speed image reconstruction through short-term plasticity for spiking cameras[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 6358-6367.
