package botkop.optimizer

import botkop.autograd.Variable
import botkop.numsca.Tensor
import botkop.{numsca => ns}

case class Adam(parameters: Seq[Variable],
                lr: Double,
                lrDecay: Double,
                beta1: Double = 0.9,
                beta2: Double = 0.999,
                epsilon: Double = 1e-8)
    extends Optimizer(parameters) {

  val ms: Seq[Tensor] = parameters.map(p => ns.zeros(p.shape: _*))
  val vs: Seq[Tensor] = parameters.map(p => ns.zeros(p.shape: _*))

  var t = 1

  def step(epoch: Int): Unit = {
    val alr = lr * (1.0 / (1.0 + lrDecay * t)) // learning rate annealing: 1/t decay

    parameters.zip(ms).zip(vs).foreach {
      case ((p, m), v) =>
        val x = p.data
        val dx = p.g

        m *= beta1
        m += (1 - beta1) * dx
        val mt = m / (1 - math.pow(beta1, t))

        v *= beta2
        v += (1 - beta2) * ns.square(dx)
        val vt = v / (1 - math.pow(beta2, t))

        x -= alr * mt / (ns.sqrt(vt) + epsilon)

        t += 1
    }
  }

}
