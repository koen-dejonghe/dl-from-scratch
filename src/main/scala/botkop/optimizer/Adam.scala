package botkop.optimizer

import botkop.autograd.Variable
import botkop.numsca.Tensor
import botkop.{numsca => ns}

case class Adam(parameters: Seq[Variable],
                lr: Double,
                beta1: Double = 0.9,
                beta2: Double = 0.999,
                epsilon: Double = 1e-8)
    extends Optimizer(parameters) {

  val expAvgs: Seq[Tensor] = parameters.map(p => ns.zeros(p.shape: _*))
  val expAvgSqs: Seq[Tensor] = parameters.map(p => ns.zeros(p.shape: _*))
  val steps: Array[Double] = Array.fill(parameters.length)(0)

  def step(epoch: Int): Unit = {
    parameters.indices.foreach { i =>
      val p = parameters(i)
      val expAvg = expAvgs(i)
      val expAvgSq = expAvgSqs(i)
      steps(i) += 1
      val step = steps(i)

      val x = p.data
      val dx = p.g

      val biasCorrection1 = 1 - math.pow(beta1, step)
      val biasCorrection2 = 1 - math.pow(beta2, step)

      expAvg *= beta1
      expAvg += (1 - beta1) * dx

      expAvgSq *= beta2
      expAvgSq += (1 - beta2) * ns.square(dx)

      val denom = (ns.sqrt(expAvgSq) / math.sqrt(biasCorrection2)) + epsilon
      val stepSize = lr / biasCorrection1

      //  p.data.addcdiv_(-step_size, exp_avg, denom)
      x -= stepSize * (expAvg / denom)
    }
  }
}

