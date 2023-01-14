# sobel_ocl

文章https://mp.weixin.qq.com/s/bI1hWEt44DfDUpEbYZ1vJw的代码

书接上文：[深度学习加速：使用opencl加速算子运算](http://mp.weixin.qq.com/s?__biz=MzIwNDgyNTc4NQ==&mid=2247484581&idx=1&sn=6e9a859a51d4b4e3b6eb1efe14b81aaa&chksm=973b7173a04cf865f7e0c19a7a670836ed57636a63f5814f696c2f18d71b98d72ea27e5760c9&scene=21#wechat_redirect)

在边缘检测中，常用的一种模板是Sobel 算子。Sobel 算子有两个，一个是检测水平边缘的 ；另一个是检测垂直边缘的 。

由于Sobel算子是滤波算子的形式，用于提取边缘，可以利用快速卷积函数， 简单有效，因此应用广泛。

算子模板：

![image](https://user-images.githubusercontent.com/20589365/212476366-6552afc4-d652-46e0-b1af-bb342126ecd5.png)

为了保证结果的正确性，我们手动实现了一版：

注意，为了方便实现，在开始前，在原来的矩阵边缘会padding下，这个长度为2.

```
void naive_impl(int *data_in, int *data_out, int *filter_in, int rows, int cols, int K)
{
  for (int ki = 0; ki < rows; ki++)
  {
    for (int kj = 0; kj < cols; kj++)
    {
      int sum = 0;
      for (int y = 0; y < K; y++)
      {
        for (int x = 0; x < K; x++)
        {
          sum = sum + filter_in[y * K + x] * data_in[Wn * (y + ki) + x + kj];
        }
      }
      data_out[ki * cols + kj] = sum;
    }
  }
}
```

实现后很容易迁移到opencl设备，根据上次的推文，很容易写出cl的代码：

```
__kernel void sobel(__global int *data_in,  // image input
                    __global int *filter,   // filter input
                    int kernel_size,        // filter kernel size
                    __global int *data_out) // feature map output
{

  int sum = 0; // multiply and sum of filter and data

  int W = get_global_size(0);     // work group global size
  int x = get_global_id(0);       // global id x
  int y = get_global_id(1);       // global id y
  int Wn = W + (kernel_size - 1); // padded image width

  for (int ki = 0; ki < kernel_size; ki++) {
    for (int kj = 0; kj < kernel_size; kj++) {
      sum += filter[ki * kernel_size + kj] * data_in[Wn * (y + ki) + x + kj];
    }
  }

  data_out[y * W + x] = sum;
}
```

是不是很容易，首先来看下加速效果：

以循环100次为例：

**naive_impl cost: 369 ms**

**opencl cost：101ms**

是不是直接快了3倍？

慢着还有，opencl的简单调优包括了：local_work_size，就是这样简单的调整就能带来收益

不同local_work_size下 ocl implement cost:

* **local_work_size: (16,16) 92ms**
* **local_work_size: (8,8) 101ms**
* **local_work_size: (8,32) 98ms**

是不是能继续快了10ms？还有，再简单使用下opencl的内置函数：比如vloadn、缓存等

在local_work_size为 (16,16)下，速度达到了 85ms

对应cl的代码：见sobel_vloadn

```
__kernel void sobel_vloadn(const __global int *data_in, // image input
                           const __global int *filter,  // filter input
                           int kernel_size,             // filter kernel size
                           __global int *data_out)// feature map output
{

  int W = get_global_size(0);     // work group global size
  int x = get_global_id(0);       // global id x
  int y = get_global_id(1);       // global id y
  int Wn = W + (kernel_size - 1); // padded image width

  int3 d1 = vload3(0, data_in + Wn * y + x);
  int3 d2 = vload3(0, data_in + Wn * (y + 1) + x);
  int3 d3 = vload3(0, data_in + Wn * (y + 2) + x);

  int3 f1 = vload3(0, filter);
  int3 f2 = vload3(0, filter + kernel_size);
  int3 f3 = vload3(0, filter + 2 * kernel_size);

  float3 df1 = convert_float3(d1);
  float3 df2 = convert_float3(d2);
  float3 df3 = convert_float3(d3);

  float3 ff1 = convert_float3(f1);
  float3 ff2 = convert_float3(f2);
  float3 ff3 = convert_float3(f3);

  float sum = dot(ff1, df1) + dot(ff2, df2) + dot(ff3, df3);

  vstore(sum, 0, data_out + y * W + x);
}
```


入门级别的加速就到这里了，大佬们可以自行看看opencl的文档

https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_C.html

https://registry.khronos.org/OpenCL/specs/opencl-1.2.pdf

后续推荐使用下面这个入门：

http://analog.nik.uni-obuda.hu/ParhuzamosProgramozasuHardver/02_GPGPU-Irodalom/03_CUDA-Irodalom_LovasIstvan/OpenCl/Books/%5BMatthew_Scarpino%5D_OpenCL_in_Action_How_to_Accele(Bookos.org).pdf

