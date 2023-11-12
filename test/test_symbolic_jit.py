import unittest
from tinygrad.jit import TinyJit
from tinygrad.helpers import getenv
from tinygrad.shape.symbolic import Variable
from tinygrad.tensor import Tensor, Device
import numpy as np

@unittest.skipIf(getenv("ARM64") or getenv("PTX"), "ARM64 and PTX are not supported")
@unittest.skipUnless(Device.DEFAULT in ["GPU", "METAL", "CLANG", "CUDA", "LLVM"], f"{Device.DEFAULT} is not supported")
class TestSymbolicJit(unittest.TestCase):
  def test_plus1(self):
    def f(a): return (a+1).realize()
    jf = TinyJit(f)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      a = Tensor.rand(3, i)
      symbolic = jf(a.reshape(3, vi)).reshape(3, i).numpy()
      expected = f(a).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert len(jf.jit_cache) == 1

  def test_add(self):
    def f(a, b): return (a+b).realize()
    jf = TinyJit(f)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      a = Tensor.rand(3, i)
      b = Tensor.rand(3, i)
      symbolic = jf(a.reshape(3, vi), b.reshape(3, vi)).reshape(3, i).numpy()
      expected = f(a, b).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert len(jf.jit_cache) == 1

  def test_matmul(self):
    def f(a, b): return (a@b).realize()
    jf = TinyJit(f)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      a = Tensor.rand(3, i)
      b = Tensor.rand(i, 5)
      symbolic = jf(a.reshape(3, vi), b.reshape(vi, 5)).numpy()
      expected = f(a, b).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert len(jf.jit_cache) == 1

  def test_mixed_with_no_symbol_kernel(self):
    def f(a, b):
      s = (a@b).realize()
      s = (s+s).realize() # this one does not have symbols in input
      return s
    jf = TinyJit(f)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      a = Tensor.rand(3, i)
      b = Tensor.rand(i, 5)
      symbolic = jf(a.reshape(3, vi), b.reshape(vi, 5)).numpy()
      expected = f(a, b).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert len(jf.jit_cache) == 2

  def test_attention(self):
    def f(q, k, v): return Tensor.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).realize()
    jf = TinyJit(f)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      q = Tensor.rand(2, 1, 4, 8)
      k = Tensor.rand(2, i, 4, 8)
      v = Tensor.rand(2, i, 4, 8)
      symbolic = jf(q, k.reshape(2, vi, 4, 8), v.reshape(2, vi, 4, 8)).reshape(2, 4, 1, 8).numpy()
      expected = f(q, k, v).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert len(jf.jit_cache) == 6

  def test_cat_dim0(self):
    def f(a, b): return a.cat(b, dim=0).realize()
    jf = TinyJit(f)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      a = Tensor.rand(i, 3)
      b = Tensor.rand(2, 3)
      symbolic = jf(a.reshape(vi, 3), b).reshape(i+2, 3).numpy()
      expected = f(a, b).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert len(jf.jit_cache) == 1

  def test_cat_dim1(self):
    def f(a, b): return a.cat(b, dim=1).realize()
    jf = TinyJit(f)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      a = Tensor.rand(3, i)
      b = Tensor.rand(3, 2)
      symbolic = jf(a.reshape(3, vi), b).reshape(3, i+2).numpy()
      expected = f(a, b).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert len(jf.jit_cache) == 1

  def test_cat_dim0_two_vars(self):
    def f(a, b): return a.cat(b, dim=0).realize()
    jf = TinyJit(f)
    for i in range(1, 5):
      for j in range(1, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        a = Tensor.rand(i, 3)
        b = Tensor.rand(j, 3)
        symbolic = jf(a.reshape(vi, 3), b.reshape(vj, 3)).reshape(i+j, 3).numpy()
        expected = f(a, b).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert len(jf.jit_cache) == 1

  def test_cat_dim1_two_vars(self):
    def f(a, b): return a.cat(b, dim=1).realize()
    jf = TinyJit(f)
    for i in range(1, 5):
      for j in range(1, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        a = Tensor.rand(3, i)
        b = Tensor.rand(3, j)
        symbolic = jf(a.reshape(3, vi), b.reshape(3, vj)).reshape(3, i+j).numpy()
        expected = f(a, b).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert len(jf.jit_cache) == 1

  def test_two_vars_plus1(self):
    def f(a, b): return (a@b+1).realize()
    jf = TinyJit(f)
    for i in range(1, 5):
      for j in range(1, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        a = Tensor.rand(i, 3)
        b = Tensor.rand(3, j)
        symbolic = jf(a.reshape(vi, 3), b.reshape(3, vj)).reshape(i, j).numpy()
        expected = f(a, b).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert len(jf.jit_cache) == 1

  def test_jit_symbolic_shape_mismatch(self):
    @TinyJit
    def add(a, b): return (a+b).realize()
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      a = Tensor.rand(3, i).reshape(3, vi)
      b = Tensor.rand(3, i).reshape(3, vi)
      c = add(a, b)
    vi2 = Variable("i", 1, 10).bind(7)
    a = Tensor.rand(3, 7).reshape(3, vi2)
    bad = Tensor.rand(4, 7).reshape(4, vi2)
    with self.assertRaises(AssertionError):
      add(a, bad)

  def test_shrink(self):
    # shrink is a movement, so we pair it with a simple function to test the JIT interaction
    def f(a): return (a+1).realize()
    jf = TinyJit(f)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      a = Tensor.rand(7, 11)
      symbolic = a.shrink(((3,5),(vi,vi+2)))
      symbolic = jf(symbolic).numpy()
      expected = f(a.shrink(((3,5),(i,i+2)))).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert len(jf.jit_cache) == 1

  def test_whisper_attention(self):
    Tensor.manual_seed(0)
    def block(qkvo_weights, x, sk=None, sv=None, jit_ctx=None):
      q, k, v = x.linear(qkvo_weights[0]), x.linear(qkvo_weights[1]), x.linear(qkvo_weights[2])
      if sk is not None:
        k,v = sk.cat(k, dim=1), sv.cat(v, dim=1)

      last_dim = q.shape[-1] // 6
      aq = q.reshape(*q.shape[:2], 6, last_dim).permute(0, 2, 1, 3)
      ak = k.reshape(*k.shape[:2], 6, last_dim).permute(0, 2, 1, 3)
      av = v.reshape(*v.shape[:2], 6, last_dim).permute(0, 2, 1, 3)
      attn = Tensor.scaled_dot_product_attention(aq, ak, av)
      wv = attn.permute(0, 2, 1, 3).flatten(start_dim=2)
      return wv.linear(qkvo_weights[3]).realize(), k.realize(), v.realize()

    num_blocks = 4

    qkvo_weights = Tensor.rand(num_blocks, 4, 384, 384).realize()
    self_attn_kv = [(None, None) for _ in range(num_blocks)]
    jit_self_attn_kv = [(None, None) for _ in range(num_blocks)]

    x = Tensor.rand(16, 2, 384).realize()

    for b in range(num_blocks):
      x, sk, sv = block(qkvo_weights[b], x)
      self_attn_kv[b] = (sk, sv)
      jit_self_attn_kv[b] = (sk, sv)

    # jit_block = [block for _ in range(num_blocks)]
    jit_block = [TinyJit(block) for _ in range(num_blocks)]
    for i in range(40):
      x = Tensor.rand(16, 1, 384).realize()
      jit_x = x
      for b in range(num_blocks):
        (sk, sv) = self_attn_kv[b]
        x, sk, sv = block(qkvo_weights[b], x, sk, sv)
        self_attn_kv[b] = (sk, sv)

        (jit_sk, jit_sv) = jit_self_attn_kv[b]
        len = jit_sk.shape[1]
        len_v = Variable("len", 1, 224).bind(len)
        jit_sk = jit_sk.reshape(jit_sk.shape[0], len_v, jit_sk.shape[2])
        jit_sv = jit_sv.reshape(jit_sv.shape[0], len_v, jit_sv.shape[2])

        jit_x, jit_sk, jit_sv = jit_block[b](qkvo_weights[b], jit_x, jit_sk, jit_sv, jit_ctx={len_v.unbind()[0]: len})

        jit_sk = jit_sk.reshape(jit_sk.shape[0], len+1, jit_sk.shape[2])
        jit_sv = jit_sv.reshape(jit_sv.shape[0], len+1, jit_sv.shape[2])

        jit_self_attn_kv[b] = (jit_sk, jit_sv)

        np.testing.assert_allclose(sk.numpy(), jit_sk.numpy(), atol=1e-6, rtol=1e-6, err_msg=f'sk mismatch i={i}, block={b}')
        np.testing.assert_allclose(sv.numpy(), jit_sv.numpy(), atol=1e-6, rtol=1e-6, err_msg=f'sv mismatch i={i}, block={b}')


if __name__ == '__main__':
  unittest.main()