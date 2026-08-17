# GPU smoke autodiff comparison

GPU results are checked against the CPU run at the same grid size.

| Backend | Grid | Forward | Gradient | Logical residual | Bounded Tape | Temporary Tape delta | GPU time / CPU time | Speedup over CPU |
| :--- | ---: | :---: | :---: | :---: | :---: | ---: | ---: | ---: |
| metal | 256 | True | True | True | True | 17409948 | 0.234 | 4.274× |
| metal | 512 | True | True | True | True | 70948764 | 0.132 | 7.600× |
| metal | 1024 | True | True | True | True | 133778876 | 0.079 | 12.600× |
| vulkan | 256 | True | True | True | True | 17409948 | 0.196 | 5.099× |
| vulkan | 512 | True | True | True | True | 70948764 | 0.091 | 10.931× |
| vulkan | 1024 | True | True | True | True | 133778876 | 0.050 | 19.836× |
