#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>

#include "CUDACore/cudaCheck.h"
#include "CUDACore/prefixScan.h"

template <typename T>
void testPrefixScan(uint32_t size, sycl::nd_item<3> item_ct1, sycl::stream stream_ct1, T *ws, T *c, T *co) {
  auto first = item_ct1.get_local_id(2);
  for (auto i = first; i < size; i += item_ct1.get_local_range().get(2))
    c[i] = 1;
  item_ct1.barrier();

  blockPrefixScan(c, co, size, item_ct1, ws);
  blockPrefixScan(c, size, ws, item_ct1);

  assert(1 == c[0]);
  assert(1 == co[0]);
  for (auto i = first + 1; i < size; i += item_ct1.get_local_range().get(2)) {
    if (c[i] != c[i - 1] + 1)
      /*
      DPCT1015:10: Output needs adjustment.
      */
      stream_ct1 << "failed %d %d %d: %d %d\n";
    assert(c[i] == c[i - 1] + 1);
    assert(c[i] == i + 1);
    assert(c[i] = co[i]);
  }
}

template <typename T>
void testWarpPrefixScan(uint32_t size, sycl::nd_item<3> item_ct1, sycl::stream stream_ct1, T *c, T *co) {
  assert(size <= 32);

  auto i = item_ct1.get_local_id(2);
  c[i] = 1;
  item_ct1.barrier();

  warpPrefixScan(c, co, i, item_ct1, 0xffffffff);
  warpPrefixScan(c, i, 0xffffffff, item_ct1);
  item_ct1.barrier();

  assert(1 == c[0]);
  assert(1 == co[0]);
  if (i != 0) {
    if (c[i] != c[i - 1] + 1)
      /*
      DPCT1015:11: Output needs adjustment.
      */
      stream_ct1 << "failed %d %d %d: %d %d\n";
    assert(c[i] == c[i - 1] + 1);
    assert(c[i] == i + 1);
    assert(c[i] = co[i]);
  }
}

void init(uint32_t *v, uint32_t val, uint32_t n, sycl::nd_item<3> item_ct1, sycl::stream stream_ct1) {
  auto i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);
  if (i < n)
    v[i] = val;
  if (i == 0)
    stream_ct1 << "init\n";
}

void verify(uint32_t const *v, uint32_t n, sycl::nd_item<3> item_ct1, sycl::stream stream_ct1) {
  auto i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);
  if (i < n)
    assert(v[i] == i + 1);
  if (i == 0)
    stream_ct1 << "verify\n";
}

int main() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::cout << "warp level" << std::endl;
  // std::cout << "warp 32" << std::endl;
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    // accessors to device memory
    sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> c_acc_ct1(sycl::range<1>(1024),
                                                                                                  cgh);
    sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> co_acc_ct1(sycl::range<1>(1024),
                                                                                                   cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)), [=](sycl::nd_item<3> item_ct1) {
          testWarpPrefixScan<int>(32, item_ct1, stream_ct1, c_acc_ct1.get_pointer(), co_acc_ct1.get_pointer());
        });
  });
  dev_ct1.queues_wait_and_throw();
  // std::cout << "warp 16" << std::endl;
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    // accessors to device memory
    sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> c_acc_ct1(sycl::range<1>(1024),
                                                                                                  cgh);
    sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> co_acc_ct1(sycl::range<1>(1024),
                                                                                                   cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)), [=](sycl::nd_item<3> item_ct1) {
          testWarpPrefixScan<int>(16, item_ct1, stream_ct1, c_acc_ct1.get_pointer(), co_acc_ct1.get_pointer());
        });
  });
  dev_ct1.queues_wait_and_throw();
  // std::cout << "warp 5" << std::endl;
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    // accessors to device memory
    sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> c_acc_ct1(sycl::range<1>(1024),
                                                                                                  cgh);
    sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> co_acc_ct1(sycl::range<1>(1024),
                                                                                                   cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)), [=](sycl::nd_item<3> item_ct1) {
          testWarpPrefixScan<int>(5, item_ct1, stream_ct1, c_acc_ct1.get_pointer(), co_acc_ct1.get_pointer());
        });
  });
  dev_ct1.queues_wait_and_throw();

  std::cout << "block level" << std::endl;
  for (int bs = 32; bs <= 1024; bs += 32) {
    // std::cout << "bs " << bs << std::endl;
    for (int j = 1; j <= 1024; ++j) {
      // std::cout << j << std::endl;
      q_ct1.submit([&](sycl::handler &cgh) {
        sycl::stream stream_ct1(64 * 1024, 80, cgh);

        // accessors to device memory
        sycl::accessor<uint16_t, 1, sycl::access::mode::read_write, sycl::access::target::local> ws_acc_ct1(
            sycl::range<1>(32), cgh);
        sycl::accessor<uint16_t, 1, sycl::access::mode::read_write, sycl::access::target::local> c_acc_ct1(
            sycl::range<1>(1024), cgh);
        sycl::accessor<uint16_t, 1, sycl::access::mode::read_write, sycl::access::target::local> co_acc_ct1(
            sycl::range<1>(1024), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, bs), sycl::range<3>(1, 1, bs)), [=](sycl::nd_item<3> item_ct1) {
              testPrefixScan<uint16_t>(
                  j, item_ct1, stream_ct1, ws_acc_ct1.get_pointer(), c_acc_ct1.get_pointer(), co_acc_ct1.get_pointer());
            });
      });
      dev_ct1.queues_wait_and_throw();
      q_ct1.submit([&](sycl::handler &cgh) {
        sycl::stream stream_ct1(64 * 1024, 80, cgh);

        // accessors to device memory
        sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> ws_acc_ct1(
            sycl::range<1>(32), cgh);
        sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> c_acc_ct1(
            sycl::range<1>(1024), cgh);
        sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> co_acc_ct1(
            sycl::range<1>(1024), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, bs), sycl::range<3>(1, 1, bs)), [=](sycl::nd_item<3> item_ct1) {
              testPrefixScan<float>(
                  j, item_ct1, stream_ct1, ws_acc_ct1.get_pointer(), c_acc_ct1.get_pointer(), co_acc_ct1.get_pointer());
            });
      });
      dev_ct1.queues_wait_and_throw();
    }
  }
  dev_ct1.queues_wait_and_throw();

  int num_items = 200;
  for (int ksize = 1; ksize < 4; ++ksize) {
    // test multiblock
    std::cout << "multiblok" << std::endl;
    // Declare, allocate, and initialize device-accessible pointers for input and output
    num_items *= 10;
    uint32_t *d_in;
    uint32_t *d_out1;
    uint32_t *d_out2;

    /*
    DPCT1003:12: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    */
    ((d_in = sycl::malloc_device<uint32_t>(num_items * sizeof(uint32_t), q_ct1), 0));
    /*
    DPCT1003:13: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    */
    ((d_out1 = sycl::malloc_device<uint32_t>(num_items * sizeof(uint32_t), q_ct1), 0));
    /*
    DPCT1003:14: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    */
    ((d_out2 = sycl::malloc_device<uint32_t>(num_items * sizeof(uint32_t), q_ct1), 0));

    auto nthreads = 256;
    auto nblocks = (num_items + nthreads - 1) / nthreads;

    q_ct1.submit([&](sycl::handler &cgh) {
      sycl::stream stream_ct1(64 * 1024, 80, cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nblocks) * sycl::range<3>(1, 1, nthreads),
                                         sycl::range<3>(1, 1, nthreads)),
                       [=](sycl::nd_item<3> item_ct1) { init(d_in, 1, num_items, item_ct1, stream_ct1); });
    });

    // the block counter
    int32_t *d_pc;
    /*
    DPCT1003:15: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    */
    cudaCheck((d_pc = sycl::malloc_device<int32_t>(1, q_ct1), 0));
    /*
    DPCT1003:16: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    */
    cudaCheck((q_ct1.memset(d_pc, 0, 4).wait(), 0));

    nthreads = 1024;
    nblocks = (num_items + nthreads - 1) / nthreads;
    q_ct1.submit([&](sycl::handler &cgh) {
      // accessors to device memory
      sycl::accessor<uint32_t, 1, sycl::access::mode::read_write, sycl::access::target::local> ws_acc_ct1(
          sycl::range<1>(32), cgh);
      sycl::accessor<bool, 0, sycl::access::mode::read_write, sycl::access::target::local> isLastBlockDone_acc_ct1(cgh);
      sycl::accessor<uint32_t, 1, sycl::access::mode::read_write, sycl::access::target::local> psum_acc_ct1(
          sycl::range<1>(1024), cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nblocks) * sycl::range<3>(1, 1, nthreads),
                                         sycl::range<3>(1, 1, nthreads)),
                       [=](sycl::nd_item<3> item_ct1) {
                         multiBlockPrefixScan(d_in,
                                              d_out1,
                                              num_items,
                                              d_pc,
                                              item_ct1,
                                              ws_acc_ct1.get_pointer(),
                                              isLastBlockDone_acc_ct1.get_pointer(),
                                              psum_acc_ct1.get_pointer());
                       });
    });
    q_ct1.submit([&](sycl::handler &cgh) {
      sycl::stream stream_ct1(64 * 1024, 80, cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nblocks) * sycl::range<3>(1, 1, nthreads),
                                         sycl::range<3>(1, 1, nthreads)),
                       [=](sycl::nd_item<3> item_ct1) { verify(d_out1, num_items, item_ct1, stream_ct1); });
    });
    dev_ct1.queues_wait_and_throw();

    // test cub
    std::cout << "cub" << std::endl;
    // Determine temporary device storage requirements for inclusive prefix sum
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out2, num_items);

    std::cout << "temp storage " << temp_storage_bytes << std::endl;

    // Allocate temporary storage for inclusive prefix sum
    // fake larger ws already available
    temp_storage_bytes *= 8;
    /*
    DPCT1003:17: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    */
    cudaCheck((d_temp_storage = (void *)sycl::malloc_device(temp_storage_bytes, q_ct1), 0));
    std::cout << "temp storage " << temp_storage_bytes << std::endl;
    // Run inclusive prefix sum
    CubDebugExit(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out2, num_items));
    std::cout << "temp storage " << temp_storage_bytes << std::endl;

    q_ct1.submit([&](sycl::handler &cgh) {
      sycl::stream stream_ct1(64 * 1024, 80, cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nblocks) * sycl::range<3>(1, 1, nthreads),
                                         sycl::range<3>(1, 1, nthreads)),
                       [=](sycl::nd_item<3> item_ct1) { verify(d_out2, num_items, item_ct1, stream_ct1); });
    });
    dev_ct1.queues_wait_and_throw();
  }  // ksize
  return 0;
}
