package botkop.module

import botkop.autograd.Variable
import botkop.{numsca => ns}

case class Dropout(p: Double = 0.5) extends Module {
  require(p > 0 && p < 1,
    s"dropout probability has to be between 0 and 1, but got $p")

  override def forward(x: Variable): Variable = if (isTraining) {
    val mask: Variable = Variable((ns.rand(x.shape: _*) < p) / p)
    x * mask
  } else {
    x
  }
}

