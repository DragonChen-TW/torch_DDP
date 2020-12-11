## EntryPoint
### Singal machine Multiple GPU
`python3 start.py`

### Multiple machine Multiple GPU
Decide your world size.

`python3 mnist_train.py 0 {size}`  
`python3 mnist_train.py 1 {size}`  
...  
`python3 mnist_train.py {size - 1} {size}`

e.g. world size is 2  
In mahcine 1:  
`python3 mnist_train.py 0 2`

In machine 2:  
`python3 mnist_train.py 1 2`

## Settings

Inside `def setup():`

### communication

`'nccl'` => other options: `'gloo'`, `'nccl'`, `'mpi'`

### init_method
IP adress to listen

Inside `def demo_basic():`

### GPU rank
you can set `gpu_rank` is a constant or a mapping results

### model
supported model: `resnet18()`, `resnet34()`, `resnet50()`

### data path
`data = get_mnist('~/data')`  
path to save saved MNIST