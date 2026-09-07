# GPU smoke autodiff comparison

GPU results are checked against the CPU run at the same grid size.

| Backend | Grid | Forward | Gradient | Logical residual | Bounded Tape | Temporary Tape delta | GPU time / CPU time | Speedup over CPU |
| :--- | ---: | :---: | :---: | :---: | :---: | ---: | ---: | ---: |
| metal | 32 | True | True | True | True | -157476 | 5.963 | 0.168× |
| metal | 64 | True | True | True | True | 679068 | 2.049 | 0.488× |
| metal | 128 | True | True | True | True | 4025244 | 0.616 | 1.623× |
| metal | 256 | True | True | True | True | 17409948 | 0.230 | 4.350× |
| metal | 512 | True | True | True | True | 70948764 | 0.133 | 7.503× |
| metal | 1024 | True | True | True | True | 133778876 | 0.078 | 12.747× |
| vulkan | 32 | True | True | True | True | -157476 | 6.180 | 0.162× |
| vulkan | 64 | True | True | True | True | 679068 | 2.032 | 0.492× |
| vulkan | 128 | True | True | True | True | 4025244 | 0.590 | 1.695× |
| vulkan | 256 | True | True | True | True | 17409948 | 0.193 | 5.174× |
| vulkan | 512 | True | True | True | True | 70948764 | 0.093 | 10.795× |
| vulkan | 1024 | True | True | True | True | 133778876 | 0.050 | 19.818× |
