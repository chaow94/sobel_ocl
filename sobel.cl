
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

__kernel void sobel_vloadn(const __global int *data_in, // image input
                           const __global int *filter,  // filter input
                           int kernel_size,             // filter kernel size
                           __global int *data_out)      // feature map output
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