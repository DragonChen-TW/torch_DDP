## EntryPoint
### Singal machine Multiple GPU
`python3 start.py`

### Multiple machine Multiple GPU
Decide your world size.

`python3 mnist_train.py 0 {size}`  
`python3 mnist_train.py 1 {size}`  
...  
`python3 mnist_train.py {size - 1} {size}`

e.g. world is 2  
In mahcine 1:  
`python3 mnist_train.py 0 {size}`

In machine 2:  
`python3 mnist_train.py 1 {size}`