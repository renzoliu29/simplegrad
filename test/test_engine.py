import torch
from simplegrad.engine import Value


def test_sanity_check_relu():

    x = Value(-4.0)
    z = 2 * x + 2 * x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xsg, ysg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 * x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass
    assert ysg.data == ypt.data.item()

    # backward pass
    assert xsg.grad == xpt.grad.item()


def test_sanity_check_tanh():

    x = Value(-4.0)
    z = 2 * x + 2 * x
    q = z.tanh() + z * x
    h = (z * z).tanh()
    y = h + q + q * x
    y.backward()
    xsg, ysg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 * x
    q = z.tanh() + z * x
    h = (z * z).tanh()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass
    assert ysg.data == ypt.data.item()

    # backward pass
    assert xsg.grad == xpt.grad.item()


def test_more_ops_relu():
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    asg, bsg, gsg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6

    # forward pass
    assert abs(gsg.data - gpt.data.item()) <= tol

    # backward pass
    assert abs(asg.grad - apt.grad.item()) <= tol
    assert abs(bsg.grad - bpt.grad.item()) <= tol


def test_more_ops_tanh():
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).tanh()
    d += 3 * d + (b - a).tanh()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    asg, bsg, gsg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).tanh()
    d += 3 * d + (b - a).tanh()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6

    # forward pass
    assert abs(gsg.data - gpt.data.item()) <= tol

    # backward pass
    assert abs(asg.grad - apt.grad.item()) <= tol
    assert abs(bsg.grad - bpt.grad.item()) <= tol
