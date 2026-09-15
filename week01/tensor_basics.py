import torch

#1.创建Tensor
a = torch.tensor([1, 2, 3])
print(a)

#2.查看形状
print("shape:", a.shape)

#3.查看数据类型
print("dtype:", a.dtype)

#4.查看所在设备
print("device:", a.device)

b = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

print(b)
print("shape:", b.shape)
print("dtype:", b.dtype)
print("device:", b.device)
print("ndim:", b.ndim)
print("number of elements:", b.numel())

print("------------------------------")

c = torch.tensor([1.0, 2.0, 3.0])
d = torch.tensor([1, 2, 3], dtype = torch.float32)

print("c:", c)
print("c dtype:", c.dtype)
print("d:", d)
print("d dtype:", d.dtype)

print("------------------------------")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("selected device:", device)
e = torch.tensor([1.0, 2.0, 3.0])
print("before:", e.device)
e = e.to(device)
print("after:", e.device)

print("------------------------------")

x = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

y = torch.tensor([
    [5.0, 6.0],
    [7.0, 8.0]
])

print("x + y:")
print(x + y)

print("x * y:")
print(x * y)

print("x @ y:")
print(x @ y)

print("------------------------------")

f = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

print("original:")
print(f)
print("original shape:", f.shape)

g = f.reshape(3, 2)

print("after reshape:")
print(g)
print("new shape:", g.shape)

h = f.reshape(6)

print("resshape to 1D:")
print(h)
print("shape:", h.shape)

print("------------------------------")

r1 = torch.rand(2, 3)
print("rand:")
print(r1)

r2 = torch.randn(2, 3)
print("randn:")
print(r2)

r3 = torch.zeros(2, 3)
print("zeros:")
print(r3)

r4 = torch.ones(2, 3)
print("ones:")
print(r4)

print("r1 shape:", r1.shape)
print("r1 dtype:", r1.dtype)
print("r1 device:", r1.device)

print("------------------------------")

cpu_tensor = torch.rand(3, 3)

gpu_tensor = cpu_tensor.to("cuda")

print("CPU tensor device:", cpu_tensor.device)
print("GPU tensor device:", gpu_tensor.device)

result = gpu_tensor @ gpu_tensor

print("result:")
print(result)
print("result device:", result.device)

print("------------------------------")
print("------------------------------")

x = torch.rand(4, 3)
print(x)
print("shape:", x.shape)
print("dtype:", x.dtype)
print("device:", x.device)
x = x.to("cuda")
print("after device:", x.device)
x2 = x.reshape(2, 6)
y = torch.rand(6, 2)
y = y.to("cuda")
result = x2 @ y
print("result:")
print(result)
print("shape:", result.shape)
print("device:", result.device)
