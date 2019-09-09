package botkop.autograd

import botkop.{numsca => ns}
import botkop.numsca.Tensor

trait Function {
  def forward(): Variable
  def backward(g: Tensor): Unit
}

/* ================================
Functions with 1 operand
 */

case class Mean(v: Variable) extends Function {
  def forward() = Variable(data = ns.mean(v.data), Some(this))
  def backward(gradOutput: Tensor): Unit = {
    val n = v.shape.product
    v.backward(gradOutput / n)
  }
}

case class Tanh(x: Variable) extends Function {
  val tanh: Tensor = ns.tanh(x.data)
  override def forward(): Variable = Variable(tanh, Some(this))
  override def backward(g: Tensor): Unit = {
    val dx = (1 - ns.square(tanh)) * g
    x.backward(dx)
  }
}

case class Exp(v: Variable) extends Function {
  val cache: Tensor = ns.exp(v.data)
  def forward() = Variable(data = cache, Some(this))
  def backward(gradOutput: Tensor): Unit = {
    v.backward(gradOutput * cache)
  }
}

case class Threshold(x: Variable, d: Double) extends Function {
  override def forward(): Variable = Variable(ns.maximum(x.data, d), Some(this))
  override def backward(gradOutput: Tensor): Unit = {
    x.backward(gradOutput * (x.data > d))
  }
}

/* ================================
Functions with 2 operands
 */

case class Add(v1: Variable, v2: Variable) extends Function {
  def forward(): Variable = Variable(v1.data + v2.data, f = Some(this))
  def backward(g: Tensor): Unit = {
    v1.backward(g * 1.0)
    v2.backward(g * 1.0)
  }
}

case class Sub(v1: Variable, v2: Variable) extends Function {
  def forward(): Variable = Variable(v1.data - v2.data, Some(this))
  def backward(gradOutput: Tensor): Unit = {
    v1.backward(gradOutput)
    v2.backward(-gradOutput)
  }
}

case class Mul(v1: Variable, v2: Variable) extends Function {
  override def forward(): Variable = Variable(v1.data * v2.data, f = Some(this))
  override def backward(g: Tensor): Unit = {
    val dv2 = v2.data * g
    val dv1 = v1.data * g
    v1.backward(dv2)
    v2.backward(dv1)
  }
}

case class Div(v1: Variable, v2: Variable) extends Function {
  override def forward(): Variable = Variable(v1.data / v2.data, Some(this))
  override def backward(gradOutput: Tensor): Unit = {
    val rv2 = 1 / v2.data
    val gv1 = gradOutput * rv2
    val gv2 = -gradOutput * v1.data * (rv2 ** 2)
    v1.backward(gv1)
    v2.backward(gv2)
  }
}

case class Dot(v1: Variable, v2: Variable) extends Function {
  val w: Tensor = v1.data
  val x: Tensor = v2.data

  override def forward(): Variable = Variable(w dot x, f = Some(this))
  override def backward(g: Tensor): Unit = {
    val dw = g dot x.T
    val dx = w.T dot g
    v1.backward(dw)
    v2.backward(dx)
  }
}

case class Max(x: Variable, y: Variable) extends Function {
  def forward(): Variable = {
    val max: Tensor = ns.maximum(x.data, y.data)
    Variable(max, Some(this))
  }
  override def backward(gradOutput: Tensor): Unit = {
    x.backward((x.data >= y.data) * gradOutput)
    y.backward((x.data <= y.data) * gradOutput)
  }
}

/* ================================
Functions with scalars
 */

case class AddScalar(v: Variable, d: Double) extends Function {
  override def forward(): Variable = Variable(v.data + d, Some(this))
  override def backward(gradOutput: Tensor): Unit = {
    v.backward(gradOutput)
  }
}

case class SubScalar(v: Variable, d: Double) extends Function {
  override def forward(): Variable = Variable(v.data - d, Some(this))
  override def backward(gradOutput: Tensor): Unit = {
    v.backward(gradOutput)
  }
}

case class MulScalar(v: Variable, d: Double) extends Function {
  override def forward(): Variable = Variable(v.data * d, Some(this))
  override def backward(gradOutput: Tensor): Unit = {
    val dv = gradOutput * d
    v.backward(dv)
  }
}

case class DivScalar(v: Variable, d: Double) extends Function {
  override def forward(): Variable = Variable(v.data / d, Some(this))
  override def backward(gradOutput: Tensor): Unit = {
    val dv = gradOutput / d
    v.backward(dv)
  }
}

case class PowScalar(v: Variable, d: Double) extends Function {
  override def forward(): Variable = Variable(ns.power(v.data, d), Some(this))
  override def backward(gradOutput: Tensor): Unit = {
    val dv = d * ns.power(v.data, d - 1) * gradOutput
    v.backward(dv)
  }
}

/* ================================
Loss functions
 */
case class SoftmaxLoss(actual: Variable, target: Variable) extends Function {
  val x: Tensor = actual.data
  val y: Tensor = target.data.T

  val shiftedLogits: Tensor = x - ns.max(x, axis = 1)
  val z: Tensor = ns.sum(ns.exp(shiftedLogits), axis = 1)
  val logProbs: Tensor = shiftedLogits - ns.log(z)
  val n: Int = x.shape.head
  val loss: Double = -ns.sum(logProbs(ns.arange(n), y)) / n

  override def forward(): Variable = Variable(Tensor(loss), Some(this))

  override def backward(gradOutput: Tensor /* not used */ ): Unit = {
    val dx = ns.exp(logProbs)
    dx(ns.arange(n), y) -= 1
    dx /= n

    actual.backward(dx)
  }
}
