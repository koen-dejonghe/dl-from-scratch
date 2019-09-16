package botkop.module

import botkop.autograd.Variable
import botkop.{numsca => ns}

case class Linear(weights: Variable, bias: Variable)
  extends Module(Seq(weights, bias)) {

  override def forward(x: Variable): Variable = {
    (x dot weights.t()) + bias
  }
}

object Linear {
  def apply(inFeatures: Int, outFeatures: Int): Linear = {
    val w = ns.randn(outFeatures, inFeatures) * math.sqrt(2.0 / outFeatures)
    val weights = Variable(w)
    val b = ns.zeros(1, outFeatures)
    val bias = Variable(b)
    Linear(weights, bias)
  }
}

